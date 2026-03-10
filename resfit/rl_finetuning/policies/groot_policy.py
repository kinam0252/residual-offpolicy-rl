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
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

try:
    from gr00t.policy.server_client import PolicyClient
except ImportError:
    PolicyClient = None  # graceful fallback for py_compile


def _bootstrap_typing_extensions_override() -> None:
    """Force-load newer typing_extensions from ISO_DEPS_DIR when available.

    This mirrors the standalone iface script behavior and avoids failures when
    Isaac prebundle ships an older typing_extensions missing newer symbols.
    """
    try:
        deps_dir = Path(
            os.environ.get(
                "ISO_DEPS_DIR",
                str(Path(__file__).resolve().parents[3] / "workspace" / ".pydeps_groot_iso"),
            )
        ).expanduser().resolve()
        te_file = deps_dir / "typing_extensions.py"
        if not te_file.exists():
            return

        deps_dir_str = str(deps_dir)
        if deps_dir_str in sys.path:
            sys.path.remove(deps_dir_str)
        sys.path.insert(0, deps_dir_str)

        spec = importlib.util.spec_from_file_location("typing_extensions", str(te_file))
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "NoExtraItems"):
            sys.modules["typing_extensions"] = module
    except Exception:
        return


_bootstrap_typing_extensions_override()

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
            self._ensure_typing_extensions_compat()
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

        self._video_modality_keys = ["wrist_view", "left_view", "right_view"]
        self._state_modality_keys = [_STATE_JOINT_KEY, _STATE_GRIPPER_KEY]
        self._language_key = _LANGUAGE_KEY
        try:
            mod_cfg = self.get_modality_config()
            self._video_modality_keys = list(getattr(mod_cfg["video"], "modality_keys", self._video_modality_keys))
            self._state_modality_keys = list(getattr(mod_cfg["state"], "modality_keys", self._state_modality_keys))
            lang_keys = list(getattr(mod_cfg["language"], "modality_keys", []))
            if len(lang_keys) > 0:
                self._language_key = str(lang_keys[0])
        except Exception:
            pass

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

    def get_modality_config(self) -> Any:
        if self.mode == "local":
            return self.local_policy.get_modality_config()
        return self.client.get_modality_config()

    def infer_action_chunk(self, obs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        info: dict[str, Any] = {}

        # Path A: prebuilt GR00T observation dict
        if "video" in obs and "state" in obs and "language" in obs:
            if self.mode == "local":
                pred_action, raw_info = self.local_policy.get_action(obs)
            else:
                pred = self.client.get_action(obs)
                pred_action = pred[0] if isinstance(pred, (tuple, list)) and len(pred) == 2 else pred
                raw_info = pred[1] if isinstance(pred, (tuple, list)) and len(pred) == 2 else {}
            if raw_info is not None:
                info = raw_info
            chunk = self._parse_action(pred_action)
            return chunk, info

        # Path B: raw IsaacLab-style single-env observation dict (batch dim=1)
        chunk = self._call_local(obs, 0) if self.mode == "local" else self._call_server(obs, 0)
        if chunk is None:
            chunk = np.zeros((self.action_horizon, 7), dtype=np.float32)
        return chunk, info

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
            language_dict = {self._language_key: [[str(lang_str)]]}

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
            language_dict = {self._language_key: [[str(lang_str)]]}

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
        def _extract_hwc(key: str) -> np.ndarray:
            img_t = obs.get(key, None)
            if img_t is None:
                return np.zeros((84, 84, 3), dtype=np.uint8)
            img = img_t[env_id].detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            return img.astype(np.uint8)

        wrist = _extract_hwc("observation.images.wrist")
        left = _extract_hwc("observation.images.back")
        right = _extract_hwc("observation.images.front")

        def _pick_video_for_key(key: str) -> np.ndarray:
            lk = str(key).lower()
            if "wrist" in lk or "ego" in lk:
                return wrist
            if "left" in lk or "back" in lk:
                return left
            if "right" in lk or "front" in lk:
                return right
            return wrist

        keys = self._video_modality_keys if len(self._video_modality_keys) > 0 else ["wrist_view", "left_view", "right_view"]
        video = {k: _pick_video_for_key(k)[None, None, ...].astype(np.uint8) for k in keys}
        if "ego_view" not in video:
            video["ego_view"] = wrist[None, None, ...].astype(np.uint8)
        return video

    def _build_state_dict(self, obs: dict[str, torch.Tensor], env_id: int) -> dict[str, np.ndarray]:
        """Extract proprioceptive state for one env."""
        joint_pos_7 = np.zeros(7, dtype=np.float32)
        gripper_frac = 0.0

        raw_joint = obs.get("observation.raw_joint_pos", None)
        raw_gripper = obs.get("observation.raw_gripper_frac", None)

        if raw_joint is not None:
            joint_np = raw_joint[env_id].detach().cpu().numpy().astype(np.float32)
            if joint_np.shape[0] >= 7:
                joint_pos_7 = joint_np[:7]
            if raw_gripper is None and joint_np.shape[0] >= 9:
                gripper_frac = float(np.clip(np.mean(joint_np[7:9]) / 0.04, 0.0, 1.0))

        if raw_gripper is not None:
            g = raw_gripper[env_id].detach().cpu().numpy().reshape(-1)
            if g.size > 0:
                gripper_frac = float(np.clip(g[0], 0.0, 1.0))

        if raw_joint is None:
            state = obs.get("observation.state", None)
            if state is not None:
                state_np = state[env_id].detach().cpu().numpy().astype(np.float32)
                if state_np.shape[0] >= 7:
                    joint_pos_7 = state_np[:7]
                if state_np.shape[0] >= 8:
                    gripper_frac = float(np.clip((state_np[7] + 1.0) / 2.0, 0.0, 1.0))

        def _pick_state_for_key(key: str) -> np.ndarray:
            lk = str(key).lower()
            if "joint" in lk:
                return joint_pos_7[None, None, :].astype(np.float32)
            if "gripper" in lk:
                return np.array([[[gripper_frac]]], dtype=np.float32)
            return np.zeros((1, 1, 1), dtype=np.float32)

        keys = self._state_modality_keys if len(self._state_modality_keys) > 0 else [_STATE_JOINT_KEY, _STATE_GRIPPER_KEY]
        state_dict = {k: _pick_state_for_key(k) for k in keys}
        if _STATE_JOINT_KEY not in state_dict:
            state_dict[_STATE_JOINT_KEY] = joint_pos_7[None, None, :].astype(np.float32)
        if _STATE_GRIPPER_KEY not in state_dict:
            state_dict[_STATE_GRIPPER_KEY] = np.array([[[gripper_frac]]], dtype=np.float32)
        return state_dict

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
    def _ensure_typing_extensions_compat():
        try:
            import typing_extensions as te
        except Exception:
            return

        class _SentinelType:
            pass

        if not hasattr(te, "NoDefault"):
            te.NoDefault = _SentinelType()
        if not hasattr(te, "NoExtraItems"):
            te.NoExtraItems = _SentinelType()

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
