"""
GR00T base-policy wrapper for resfit with action-chunk caching.

Design choices
--------------
* GR00T returns 16-token action chunks.  We cache the full chunk and serve one
  token per ``select_action`` call, advancing an internal index.
* A fresh server call happens only when the cache is exhausted (every 16 calls).
* For multi-env, each environment has its own independent cache.
* Supports both server mode (PolicyClient) and local mode (Gr00tPolicy).
* Image observations are converted to the format GR00T expects:
  ``(1, 1, H, W, 3)`` uint8 numpy arrays keyed by view name.

Usage
-----
::

    from resfit.rl_finetuning.policies.groot_policy import GR00TBasePolicy
    policy = GR00TBasePolicy(host="127.0.0.1", port=5555, num_envs=1, device="cuda:0")
    base_action = policy.select_action(obs)   # (num_envs, 7)
    policy.reset(env_ids=torch.tensor([0]))   # clear cache for env 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

try:
    from gr00t.policy.server_client import PolicyClient
except ImportError:
    PolicyClient = None  # graceful fallback for py_compile

try:
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
except ImportError:
    Gr00tPolicy = None
    EmbodimentTag = None


# ── Modality keys (same as standalone GR00T script) ──
_STATE_JOINT_KEY = "proprio.joint_pos"
_STATE_GRIPPER_KEY = "proprio.gripper_pos"
_LANGUAGE_KEY = "annotation.human.action.task_description"


@dataclass
class _FakeImageFeaturesConfig:
    """Minimal stand-in so ``BasePolicyVecEnvWrapper`` can read ``policy.config.image_features``."""
    image_features: dict = field(default_factory=lambda: {
        "observation.images.front": {},
        "observation.images.back": {},
        "observation.images.wrist": {},
    })


class GR00TBasePolicy:
    """Wraps GR00T PolicyClient to match resfit's base-policy interface.

    Attributes
    ----------
    config : object
        Exposes ``config.image_features`` so ``BasePolicyVecEnvWrapper`` can
        discover which camera keys exist.
    """

    # Mapping: resfit obs key -> GR00T video dict key
    VIEW_MAP = {
        "observation.images.wrist": "wrist_view",
        "observation.images.back": "left_view",
        "observation.images.front": "right_view",
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        num_envs: int = 1,
        device: str = "cuda:0",
        task_description: str = "Pick up the white object.",
        language_override: str | None = None,
        action_horizon: int = 16,
        model_path: str | None = None,
        embodiment_tag: str = "new_embodiment",
        strict: bool = False,
    ):
        self.host = host
        self.port = port
        self.num_envs = num_envs
        self.device = device
        self.task_description = task_description
        self.language_override = language_override
        self.action_horizon = action_horizon
        self.mode = "local" if model_path else "server"
        self.client = None
        self.local_policy = None

        if self.mode == "local":
            if Gr00tPolicy is None or EmbodimentTag is None:
                raise ImportError("gr00t local policy not found. Add Isaac-GR00T to PYTHONPATH.")
            resolved_tag = self._resolve_embodiment_tag(embodiment_tag)
            self.local_policy = Gr00tPolicy(
                embodiment_tag=resolved_tag,
                model_path=model_path,
                device=device,
                strict=strict,
            )
            try:
                mod_cfg = self.local_policy.get_modality_config()
                self.action_horizon = len(mod_cfg["action"].delta_indices)
            except Exception:
                pass
            print(f"[GR00TBasePolicy] Local mode loaded: {model_path}")
        else:
            if PolicyClient is None:
                raise ImportError("gr00t server client not found. Add Isaac-GR00T to PYTHONPATH.")
            self.client = PolicyClient(host=host, port=port, strict=False)
            if not self.client.ping():
                raise RuntimeError(f"GR00T server ping failed at {host}:{port}")
            print(f"[GR00TBasePolicy] Connected to {host}:{port}")

            try:
                mod_cfg = self.client.get_modality_config()
                self.action_horizon = len(mod_cfg["action"].delta_indices)
            except Exception:
                pass
            print(f"[GR00TBasePolicy] action_horizon={self.action_horizon}")

        # Per-env action chunk cache
        self._cached_chunks: list[np.ndarray | None] = [None] * num_envs  # (H, 7)
        self._chunk_idx: list[int] = [0] * num_envs

        # Fake config for BasePolicyVecEnvWrapper compatibility
        self.config = _FakeImageFeaturesConfig()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return base action for all envs: ``(num_envs, 7)``.

        Serves cached action-chunk tokens.  Calls GR00T server only when the
        cache for an env is exhausted.
        """
        actions = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)

        for env_id in range(self.num_envs):
            # Check if we need a fresh inference
            if self._cached_chunks[env_id] is None or self._chunk_idx[env_id] >= self._cached_chunks[env_id].shape[0]:
                chunk = self._call_local(obs, env_id) if self.mode == "local" else self._call_server(obs, env_id)
                if chunk is not None:
                    self._cached_chunks[env_id] = chunk
                    self._chunk_idx[env_id] = 0
                else:
                    # Server call failed — return zeros
                    continue

            # Serve current token
            token = self._cached_chunks[env_id][self._chunk_idx[env_id]]
            actions[env_id] = torch.tensor(token, device=self.device, dtype=torch.float32)
            self._chunk_idx[env_id] += 1

        return actions

    def reset(self, env_ids: torch.Tensor | None = None):
        """Clear action chunk cache for the specified environments."""
        if env_ids is None:
            ids = range(self.num_envs)
        else:
            ids = env_ids.cpu().numpy().tolist() if isinstance(env_ids, torch.Tensor) else env_ids
        for eid in ids:
            self._cached_chunks[eid] = None
            self._chunk_idx[eid] = 0

    # ------------------------------------------------------------------
    # Server call
    # ------------------------------------------------------------------

    def _call_server(self, obs: dict[str, torch.Tensor], env_id: int) -> np.ndarray | None:
        """Build GR00T obs dict for one env and call the server.

        Returns action chunk ``(H, 7)`` or ``None`` on failure.
        """
        try:
            video_dict = self._build_video_dict(obs, env_id)
            state_dict = self._build_state_dict(obs, env_id)
            lang_str = self.language_override if self.language_override is not None else self.task_description
            language_dict = {_LANGUAGE_KEY: [[str(lang_str)]]}

            groot_obs = {"video": video_dict, "state": state_dict, "language": language_dict}
            pred = self.client.get_action(groot_obs)
            pred_action = pred[0] if isinstance(pred, (tuple, list)) and len(pred) == 2 else pred

            chunk = self._parse_action(pred_action)
            if chunk is not None and chunk.ndim == 2 and chunk.shape[1] == 7:
                num_tokens = min(self.action_horizon, chunk.shape[0])
                return chunk[:num_tokens].astype(np.float32)
            return None
        except Exception as e:
            print(f"[GR00TBasePolicy] Server call failed for env {env_id}: {e}")
            return None

    def _call_local(self, obs: dict[str, torch.Tensor], env_id: int) -> np.ndarray | None:
        """Build GR00T obs dict for one env and call local in-process model."""
        try:
            video_dict = self._build_video_dict(obs, env_id)
            state_dict = self._build_state_dict(obs, env_id)
            lang_str = self.language_override if self.language_override is not None else self.task_description
            language_dict = {_LANGUAGE_KEY: [[str(lang_str)]]}

            groot_obs = {"video": video_dict, "state": state_dict, "language": language_dict}
            pred_action, _info = self.local_policy.get_action(groot_obs)

            chunk = self._parse_action(pred_action)
            if chunk is not None and chunk.ndim == 2 and chunk.shape[1] == 7:
                num_tokens = min(self.action_horizon, chunk.shape[0])
                return chunk[:num_tokens].astype(np.float32)
            return None
        except Exception as e:
            print(f"[GR00TBasePolicy] Local inference failed for env {env_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Observation building helpers
    # ------------------------------------------------------------------

    def _build_video_dict(self, obs: dict[str, torch.Tensor], env_id: int) -> dict[str, np.ndarray]:
        """Extract camera images for one env and format for GR00T."""
        video = {}
        for resfit_key, groot_key in self.VIEW_MAP.items():
            img_t = obs.get(resfit_key, None)
            if img_t is not None:
                # img_t: (num_envs, C, H, W) uint8
                img = img_t[env_id].detach().cpu().numpy()  # (C, H, W)
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                # (C, H, W) -> (H, W, C)
                img = np.transpose(img, (1, 2, 0))
                # GR00T expects (1, 1, H, W, C)
                video[groot_key] = img[None, None, ...].astype(np.uint8)
            else:
                # Fallback: 84x84 black
                video[groot_key] = np.zeros((1, 1, 84, 84, 3), dtype=np.uint8)

        # Alias ego_view -> wrist_view
        if "wrist_view" in video:
            video["ego_view"] = video["wrist_view"]
        return video

    def _build_state_dict(self, obs: dict[str, torch.Tensor], env_id: int) -> dict[str, np.ndarray]:
        """Extract proprioceptive state for one env."""
        state = obs.get("observation.state", None)
        if state is not None:
            state_np = state[env_id].detach().cpu().numpy().astype(np.float32)
            # First 9 dims are joint_pos_scaled, but GR00T wants raw 7 arm joints
            # We'll pass the first 7 as joint_pos (arm) and derive gripper from dim 7/8
            joint_pos_7 = state_np[:7]  # arm joints (scaled)
            # Gripper: average of two finger joints (dims 7,8), rescale to [0,1]
            gripper_frac = float(np.clip((state_np[7] + 1.0) / 2.0, 0.0, 1.0))
        else:
            joint_pos_7 = np.zeros(7, dtype=np.float32)
            gripper_frac = 0.0

        return {
            _STATE_JOINT_KEY: joint_pos_7[None, None, :].astype(np.float32),
            _STATE_GRIPPER_KEY: np.array([[[gripper_frac]]], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(pred_action: Any) -> np.ndarray | None:
        """Parse GR00T server response into ``(H, 7)`` numpy array."""
        if isinstance(pred_action, dict):
            if "action" in pred_action:
                arr = np.array(pred_action["action"], dtype=np.float32)
                return arr[0] if arr.ndim == 3 else arr
            elif "action.ee_delta" in pred_action and "action.gripper_pos" in pred_action:
                ee = np.array(pred_action["action.ee_delta"], dtype=np.float32)
                gr = np.array(pred_action["action.gripper_pos"], dtype=np.float32)
                ee = ee[0] if ee.ndim == 3 else ee
                gr = gr[0] if gr.ndim == 3 else gr
                return np.concatenate([ee, gr], axis=-1)
            return None
        arr = np.array(pred_action, dtype=np.float32)
        return arr[0] if arr.ndim == 3 else arr

    @staticmethod
    def _resolve_embodiment_tag(tag_str: str):
        if EmbodimentTag is None:
            raise ImportError("EmbodimentTag is unavailable")
        if hasattr(EmbodimentTag, str(tag_str)):
            return getattr(EmbodimentTag, str(tag_str))
        for member in EmbodimentTag:
            if str(member.value) == str(tag_str):
                return member
        valid = ", ".join(sorted(str(m.value) for m in EmbodimentTag))
        raise ValueError(f"Unknown embodiment_tag='{tag_str}'. Valid: {valid}")
