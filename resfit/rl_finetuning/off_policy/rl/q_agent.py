# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy import common_utils
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.networks.encoder import VitEncoder
from resfit.rl_finetuning.off_policy.rl.actor import Actor
from resfit.rl_finetuning.off_policy.rl.critic import Critic


class QAgent(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        prop_shape: tuple[int],
        action_dim: int,
        rl_cameras: list[str] | str,
        cfg: QAgentConfig,
        residual_actor: bool = False,
        vlm_latent_dim: int = 0,
        phase_probe_mode: bool = True,
        object_state_dim: int = 0,
        asymmetric_critic: bool = False,
    ):
        """Initialize the Q-agent.

        Parameters
        ----------
        obs_shape : tuple[int, int, int]
            Shape (C, H, W) for **a single camera** image.  When multiple
            cameras are used the same shape is assumed for every view.
        prop_shape : tuple[int]
            Shape of the proprioceptive (low-dimensional) observation vector.
        action_dim : int
            Number of action dimensions.
        rl_cameras : list[str] | str
            Name(s) of the camera images to be used by the RL policy.
            These are keys into the env's observation. A single string
            is accepted for backwards-compatibility but the preferred interface
            is to pass a list of camera names.
        cfg : QAgentConfig
            Hyper-parameter configuration dataclass.
        """
        super().__init__()
        # Normalise *rl_cameras* to a list for unified processing
        if isinstance(rl_cameras, str):
            rl_cameras = [rl_cameras]
        # rl_cameras can be empty for state-only mode

        self.rl_cameras = rl_cameras
        self.cfg = cfg
        self.residual_actor = residual_actor
        self.asymmetric_critic = asymmetric_critic  # actor=state-only, critic=depth
        self.state_only = (len(rl_cameras) == 0) and (not asymmetric_critic)

        if self.state_only:
            # State-only mode: no encoders, repr_dim=0
            self.encoders = nn.ModuleList()
            repr_dim = 0
            patch_repr_dim = 1  # dummy, not used
            print("[QAgent] State-only mode: no image encoders")
            print(f"[QAgent]   prop_dim={prop_shape[0]}, object_state_dim={object_state_dim}, "
                  f"vlm_latent_dim={vlm_latent_dim}")
        else:
            # Build the per-camera encoders (used by critic, and actor if not asymmetric)
            self.encoders: nn.ModuleList = self._build_encoders(obs_shape)
            sample_encoder = self.encoders[0]
            repr_dim_single = int(sample_encoder.repr_dim)
            patch_repr_dim = int(sample_encoder.patch_repr_dim)
            repr_dim = repr_dim_single * len(self.rl_cameras)
            print("encoder output dim: ", repr_dim)
            print("patch output dim: ", patch_repr_dim)
            if asymmetric_critic:
                print(f"[QAgent] ASYMMETRIC mode: critic uses {len(self.rl_cameras)} cameras (repr_dim={repr_dim}), actor is state-only")
            else:
                print(f"[QAgent] Image mode: {len(self.rl_cameras)} cameras, repr_dim={repr_dim}")

        assert len(prop_shape) == 1
        prop_dim = prop_shape[0] if cfg.use_prop else 0

        # VLM latent handling: two modes
        #   1. phase_probe_mode=True (default): frozen probe 2048→3 softmax
        #   2. phase_probe_mode=False: learnable projector 2048→128 (v20a compat)
        self.vlm_latent_dim = vlm_latent_dim
        self.phase_probe_mode = phase_probe_mode
        if vlm_latent_dim > 0:
            if phase_probe_mode:
                self.vlm_projected_dim = 3
                self.vlm_projector = nn.Linear(vlm_latent_dim, 3, bias=True)
                self.vlm_projector.requires_grad_(False)
                print(f"VLM phase probe: {vlm_latent_dim}D -> 3D softmax (frozen)")
            else:
                self.vlm_projected_dim = 128
                self.vlm_projector = nn.Sequential(
                    nn.Linear(vlm_latent_dim, 128),
                    nn.LayerNorm(128),
                )
                print(f"VLM projector: {vlm_latent_dim}D -> 128D (learnable, in critic_opt)")
        else:
            self.vlm_projected_dim = 0
            self.vlm_projector = None

        # Object state (cube 6D pose) — concatenated directly to prop
        self.object_state_dim = object_state_dim
        if object_state_dim > 0:
            print(f"Object state: {object_state_dim}D will be concatenated to prop")

        # Total prop dim seen by actor/critic includes projected VLM + object state
        total_prop_dim = prop_dim + self.vlm_projected_dim + self.object_state_dim
        print(f"[QAgent] total_prop_dim={total_prop_dim} "
              f"(prop={prop_dim} + vlm_proj={self.vlm_projected_dim} + obj_state={self.object_state_dim})")
        print(f"[QAgent] repr_dim={repr_dim}, patch_repr_dim={patch_repr_dim}")

        # create critics & actor (with extended prop_dim)
        # For asymmetric mode: critic uses encoder features, actor is state-only
        if self.asymmetric_critic:
            actor_repr_dim = 0  # state-only actor
            actor_patch_repr_dim = 1  # dummy
            critic_repr_dim = repr_dim  # critic uses depth encoder
            critic_patch_repr_dim = patch_repr_dim
            print(f"[QAgent] Asymmetric: actor repr_dim=0 (state-only), critic repr_dim={critic_repr_dim}")
        else:
            actor_repr_dim = repr_dim
            actor_patch_repr_dim = patch_repr_dim
            critic_repr_dim = repr_dim
            critic_patch_repr_dim = patch_repr_dim

        self.critic = Critic(
            repr_dim=critic_repr_dim,
            patch_repr_dim=critic_patch_repr_dim,
            prop_dim=total_prop_dim,
            action_dim=action_dim,
            cfg=self.cfg.critic,
        )
        self.actor = Actor(actor_repr_dim, actor_patch_repr_dim, total_prop_dim, action_dim, cfg.actor, residual_actor=residual_actor)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        print(common_utils.wrap_ruler("encoder weights"))
        print(self.encoders)
        common_utils.count_parameters(self.encoders)

        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        # optimizers
        # Freeze encoder parameters if requested
        if getattr(self.cfg, "freeze_encoder", False):
            for param in self.encoders.parameters():
                param.requires_grad = False
            print("🧊 Encoder parameters frozen - no gradient updates will be performed")

        # Create optimizers (PyTorch will ignore frozen parameters)
        if self.state_only:
            # State-only: no encoder params — use dummy param to avoid empty optimizer error
            self._encoder_dummy_param = nn.Parameter(torch.zeros(1))
            self.encoder_opt = torch.optim.AdamW([self._encoder_dummy_param], lr=self.cfg.critic_lr)
            print("[QAgent] State-only: encoder_opt uses dummy param (no-op)")
        else:
            self.encoder_opt = torch.optim.AdamW(self.encoders.parameters(), lr=self.cfg.critic_lr)
            if self.asymmetric_critic:
                print(f"[QAgent] Asymmetric: encoder trains for critic only "
                      f"({sum(p.numel() for p in self.encoders.parameters())} params)")
        critic_params = list(self.critic.parameters())
        # Include learnable VLM projector in critic_opt (only if not frozen phase probe)
        if self.vlm_projector is not None and not self.phase_probe_mode:
            critic_params += list(self.vlm_projector.parameters())
        self.critic_opt = torch.optim.AdamW(critic_params, lr=self.cfg.critic_lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.actor_lr)

        # LR schedulers for warmup (if warmup is enabled)
        self.encoder_scheduler = None
        self.critic_scheduler = None
        self.actor_scheduler = None

        if self.cfg.lr_warmup_steps > 0:
            # LinearLR scheduler that linearly ramps from start_factor to 1.0 over total_iters steps
            # Note: start_factor must be > 0 for LinearLR scheduler
            warmup_start = self.cfg.lr_warmup_start

            # Calculate start factors for each optimizer
            critic_start_factor = warmup_start / self.cfg.critic_lr if self.cfg.critic_lr > 0 else 1e-8
            critic_start_factor = max(critic_start_factor, 1e-8)

            actor_start_factor = warmup_start / self.cfg.actor_lr if self.cfg.actor_lr > 0 else 1e-8
            actor_start_factor = max(actor_start_factor, 1e-8)

            # Create schedulers with appropriate start factors
            self.encoder_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.encoder_opt, start_factor=critic_start_factor, total_iters=self.cfg.lr_warmup_steps
            )
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_opt, start_factor=critic_start_factor, total_iters=self.cfg.lr_warmup_steps
            )
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_opt, start_factor=actor_start_factor, total_iters=self.cfg.lr_warmup_steps
            )

        # data augmentation
        self.aug = common_utils.RandomShiftsAug(pad=4)

        self.bc_policies: list[nn.Module] = []
        # to log rl vs bc during evaluation
        self.stats: common_utils.MultiCounter | None = None

        self.critic_target.train(False)
        self.train(True)
        self.to(self.cfg.device)

    @staticmethod
    def load_state_dict_compat(module: nn.Module, old_sd: dict, strict: bool = False):
        """Load state_dict with backward compatibility for prop_dim changes.

        When prop_dim grows (e.g. 9→10 for adding contact force), the first
        linear layers that take prop as input will have a smaller in_features
        in the old checkpoint. This function zero-pads those weight columns
        so old checkpoints can be loaded into the new architecture.
        """
        new_sd = module.state_dict()
        padded_keys = []
        for key in old_sd:
            if key not in new_sd:
                continue
            old_w = old_sd[key]
            new_w = new_sd[key]
            if old_w.shape == new_w.shape:
                new_sd[key] = old_w
            elif old_w.dim() >= 2 and new_w.dim() >= 2 and new_w.shape[-1] > old_w.shape[-1]:
                # in_features grew — zero-pad the new columns
                new_sd[key] = torch.zeros_like(new_w)
                new_sd[key][..., :old_w.shape[-1]] = old_w
                padded_keys.append(f"{key}: {list(old_w.shape)} → {list(new_w.shape)}")
            else:
                # Other dim mismatch — just try to use old weights
                new_sd[key] = old_w
                padded_keys.append(f"{key}: SHAPE MISMATCH {list(old_w.shape)} vs {list(new_w.shape)}")
        if padded_keys:
            print(f"[load_state_dict_compat] Padded {len(padded_keys)} keys:")
            for k in padded_keys:
                print(f"  {k}")
        module.load_state_dict(new_sd, strict=False)
        return padded_keys

    def load_checkpoint_compat(self, ckpt: dict):
        """Load a full agent checkpoint with backward-compatible prop_dim handling.

        Loads: encoders, actor, actor_target, critic, critic_target, vlm_projector.
        Skips optimizer state dicts (fresh optimizers for resumed training).
        """
        padded = []
        padded += self.load_state_dict_compat(self.encoders, ckpt["encoders"])
        padded += self.load_state_dict_compat(self.actor, ckpt["actor"])
        padded += self.load_state_dict_compat(self.critic, ckpt["critic"])
        padded += self.load_state_dict_compat(self.critic_target, ckpt["critic_target"])
        if "actor_target" in ckpt:
            padded += self.load_state_dict_compat(self.actor_target, ckpt["actor_target"])

        # ── VLM projector / phase probe ──
        # If vlm_projector is frozen (phase probe), skip loading from checkpoint
        if self.vlm_projector is not None and not any(p.requires_grad for p in self.vlm_projector.parameters()):
            print("[load_checkpoint_compat] vlm_projector is frozen (phase probe) — keeping current weights")
        elif self.vlm_projector is not None and "vlm_projector" in ckpt and ckpt["vlm_projector"] is not None:
            old_sd = ckpt["vlm_projector"]
            new_sd = self.vlm_projector.state_dict()
            shapes_match = all(
                k in old_sd and old_sd[k].shape == new_sd[k].shape
                for k in new_sd
            )
            if shapes_match:
                self.vlm_projector.load_state_dict(old_sd)
                print("[load_checkpoint_compat] vlm_projector loaded (exact match)")
            else:
                print("[load_checkpoint_compat] vlm_projector shape mismatch — using fresh init")
                for k in new_sd:
                    old_shape = list(old_sd[k].shape) if k in old_sd else "MISSING"
                    print(f"  {k}: old={old_shape} new={list(new_sd[k].shape)}")
        elif self.vlm_projector is not None:
            print("[load_checkpoint_compat] WARNING: vlm_projector exists but not in checkpoint — using fresh init")

        if padded:
            print(f"[load_checkpoint_compat] Total padded layers: {len(padded)}")
        else:
            print("[load_checkpoint_compat] Exact match — no padding needed")
        return padded

    def load_phase_probe(self, probe_path: str):
        """Load pre-trained phase probe weights and freeze them."""
        import torch as _torch
        probe_ckpt = _torch.load(probe_path, map_location="cpu")
        assert self.vlm_projector is not None, "VLM projector not initialized"
        assert probe_ckpt["weight"].shape == (3, self.vlm_latent_dim), (
            f"Probe weight shape {probe_ckpt['weight'].shape} != expected (3, {self.vlm_latent_dim})"
        )
        self.vlm_projector.weight.data.copy_(probe_ckpt["weight"])
        self.vlm_projector.bias.data.copy_(probe_ckpt["bias"])
        self.vlm_projector.requires_grad_(False)
        print(f"[QAgent] Phase probe loaded from {probe_path} "
              f"(accuracy={probe_ckpt.get('test_accuracy', 0)*100:.1f}%, frozen)")

    def _build_encoders(self, obs_shape):
        """Constructs and returns an ``nn.ModuleList`` with one encoder per
        camera based on ``self.cfg.enc_type``.  All encoders share the same
        architecture and therefore yield feature tensors with identical
        dimensions which simplifies feature fusion downstream.

        For depth cameras (name starts with 'observation.depth'), uses
        in_channels=1 instead of the default obs_shape channels.
        """

        encoders = nn.ModuleList()

        for cam_name in self.rl_cameras:
            # Determine obs_shape for this camera
            if cam_name.startswith("observation.depth"):
                cam_obs_shape = (1, obs_shape[1], obs_shape[2])  # 1-channel depth
                print(f"[QAgent] Building DEPTH encoder for '{cam_name}': shape={cam_obs_shape}")
            else:
                cam_obs_shape = obs_shape  # original (3, H, W) RGB
                print(f"[QAgent] Building RGB encoder for '{cam_name}': shape={cam_obs_shape}")

            if self.cfg.enc_type == "vit":
                enc = VitEncoder(cam_obs_shape, self.cfg.vit).to(self.cfg.device)
            else:
                raise AssertionError(f"Unknown encoder type {self.cfg.enc_type}.")

            encoders.append(enc)

        print(f"[QAgent] Total encoders: {len(encoders)}, rl_cameras: {self.rl_cameras}")
        return encoders

    def add_bc_policy(self, bc_policy):
        bc_policy.train(False)
        self.bc_policies.append(bc_policy)

    def set_stats(self, stats):
        self.stats = stats

    def train(self, training=True):
        self.training = training
        self.encoders.train(training)
        self.actor.train(training)
        self.critic.train(training)

        assert not self.critic_target.training
        for bc_policy in self.bc_policies:
            assert not bc_policy.training

    def _min_over_random_two(self, q_values: torch.Tensor) -> torch.Tensor:
        """Compute min over a random subset of 2 heads from q_values [K, B, 1]."""
        assert q_values.dim() == 3 and q_values.size(-1) == 1
        num_heads = q_values.size(0)
        if num_heads <= 2:
            return torch.min(q_values[0], q_values[1])
        # Sample two unique head indices uniformly at random
        idx = torch.randperm(num_heads, device=q_values.device)[:2]
        subset = q_values.index_select(dim=0, index=idx)  # [2, B, 1]
        return subset.min(dim=0).values

    @contextmanager
    def override_act_method(self, override_method: str):
        original_method = self.cfg.act_method
        assert original_method != override_method

        self.cfg.act_method = override_method
        yield

        self.cfg.act_method = original_method

    def _prepare_prop(self, obs: dict[str, torch.Tensor], detach_vlm: bool = False) -> None:
        """Prepare observation.state by concatenating VLM output (if available).

        Phase probe mode: frozen probe → softmax 3D, always detached.
        Learnable mode: 128D projector, detach controlled by detach_vlm.
        """
        if self.vlm_projector is not None and "observation.vlm_latent" in obs:
            vlm_raw = obs["observation.vlm_latent"]  # (B, 2048)
            # ── VLM input validation ──
            assert vlm_raw.dim() == 2, f"_prepare_prop: vlm_latent must be 2D, got {vlm_raw.shape}"
            assert vlm_raw.shape[-1] == self.vlm_latent_dim, (
                f"_prepare_prop: vlm_latent dim {vlm_raw.shape[-1]} != expected {self.vlm_latent_dim}"
            )
            assert not torch.isnan(vlm_raw).any(), "_prepare_prop: NaN in vlm_latent input"
            assert not torch.isinf(vlm_raw).any(), "_prepare_prop: Inf in vlm_latent input"

            if self.phase_probe_mode:
                with torch.no_grad():
                    logits = self.vlm_projector(vlm_raw)  # (B, 3)
                    vlm_proj = torch.softmax(logits, dim=-1)  # (B, 3)
                vlm_proj = vlm_proj.detach()
            else:
                vlm_proj = self.vlm_projector(vlm_raw)  # (B, 128)
                if detach_vlm:
                    vlm_proj = vlm_proj.detach()

            # ── VLM projection validation ──
            assert vlm_proj.shape == (vlm_raw.shape[0], self.vlm_projected_dim), (
                f"_prepare_prop: vlm_proj shape {vlm_proj.shape} != expected (B, {self.vlm_projected_dim})"
            )

            state = obs["observation.state"]  # (B, 10)

            # ── State pre-concat validation ──
            assert state.shape[-1] == 10, (
                f"_prepare_prop: state should be 10D before VLM concat, got {state.shape[-1]}. "
                f"Was _prepare_prop called twice?"
            )

            obs["observation.state"] = torch.cat([state, vlm_proj], dim=-1)  # (B, 10+vlm_projected_dim)

            # ── State post-concat validation ──
            assert obs["observation.state"].shape[-1] == 10 + self.vlm_projected_dim, (
                f"_prepare_prop: state after VLM concat = {obs['observation.state'].shape[-1]} "
                f"!= {10 + self.vlm_projected_dim}"
            )

        # ── Object state concatenation (cube 6D pose) ──
        if self.object_state_dim > 0 and "observation.object_state" in obs:
            obj_state = obs["observation.object_state"]  # (B, 7)
            assert obj_state.dim() == 2, f"_prepare_prop: object_state must be 2D, got {obj_state.shape}"
            assert obj_state.shape[-1] == self.object_state_dim, (
                f"_prepare_prop: object_state dim {obj_state.shape[-1]} != {self.object_state_dim}"
            )
            state_before = obs["observation.state"].shape[-1]
            obs["observation.state"] = torch.cat([obs["observation.state"], obj_state], dim=-1)
            if not hasattr(self, '_obj_state_concat_logged'):
                print(f"[QAgent._prepare_prop] Object state concat: "
                      f"{state_before}D + {self.object_state_dim}D = "
                      f"{obs['observation.state'].shape[-1]}D")
                self._obj_state_concat_logged = True

    def _encode(self, obs: dict[str, torch.Tensor], augment: bool) -> torch.Tensor:
        r"""Encode observation images into feature tensor.

        In state-only mode (no cameras), returns a dummy empty tensor.
        """
        if self.state_only:
            # State-only mode: return empty feat (B, 0, 1)
            B = obs["observation.state"].shape[0]
            device = obs["observation.state"].device
            if not hasattr(self, '_encode_state_only_logged'):
                print(f"[QAgent._encode] State-only: returning empty feat (B={B}, 0, 1) on {device}")
                self._encode_state_only_logged = True
            return torch.zeros(B, 0, 1, device=device)

        # ── Strict input validation ──
        for cam_name in self.rl_cameras:
            assert cam_name in obs, f"QAgent._encode: missing camera '{cam_name}' in obs. Available keys: {list(obs.keys())}"
            img = obs[cam_name]
            assert img.dim() == 4, f"QAgent._encode: camera '{cam_name}' must be 4D (B,C,H,W), got {img.shape}"
            expected_c = 1 if cam_name.startswith("observation.depth") else 3
            assert img.shape[1] == expected_c, f"QAgent._encode: camera '{cam_name}' C must be {expected_c}, got {img.shape[1]}"
        assert "observation.state" in obs, "QAgent._encode: missing 'observation.state'"
        assert "observation.base_action" in obs, "QAgent._encode: missing 'observation.base_action'"

        feats = []
        for cam_idx, cam_name in enumerate(self.rl_cameras):
            data = obs[cam_name]

            if data.dtype == torch.uint8:
                # uint8 → float32 in [0,1] (RGB images)
                data = data.float().div_(255.0)
            else:
                data = data.float()
                # Depth images are already normalized [0,1] — skip /255

            if augment and data.shape[1] == 3:  # only augment RGB, not depth
                data = self.aug(data)

            # Debug: log first forward pass per camera
            if not hasattr(self, '_encode_debug_done'):
                self._encode_debug_done = set()
            if cam_name not in self._encode_debug_done:
                print(f"[QAgent._encode] cam='{cam_name}' input: shape={data.shape} "
                      f"dtype={data.dtype} range=[{data.min().item():.4f}, {data.max().item():.4f}] "
                      f"encoder_in_channels={self.encoders[cam_idx].obs_shape[0]}")
                self._encode_debug_done.add(cam_name)

            # Forward pass through the *corresponding* encoder
            feat_cam = self.encoders[cam_idx].forward(data, flatten=False)
            feats.append(feat_cam)

        # Concatenate along the *patch* dimension (dim=1)
        feat_all = torch.cat(feats, dim=1)
        return feat_all  # noqa: RET504

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if self.state_only:
            # State-only: check state dim instead of camera dim
            if obs["observation.state"].dim() == 1:
                should_unsqueeze = True
        elif obs[self.rl_cameras[0]].dim() == 3:
            should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        # ── Strict input validation ──
        if not self.state_only and not self.asymmetric_critic:
            for cam_name in self.rl_cameras:
                assert cam_name in obs, f"QAgent.act: missing camera '{cam_name}' in obs"
        assert "observation.state" in obs, "QAgent.act: missing 'observation.state'"
        assert "observation.base_action" in obs, "QAgent.act: missing 'observation.base_action'"
        # Make a shallow copy of the observation dict
        obs = copy.copy(obs)
        unsqueezed = self._maybe_unsqueeze_(obs)

        assert "feat" not in obs
        if self.asymmetric_critic:
            # Actor is state-only: skip encoding, use empty feat
            B = obs["observation.state"].shape[0]
            obs["feat"] = torch.zeros(B, 0, 1, device=obs["observation.state"].device)
        else:
            obs["feat"] = self._encode(obs, augment=False)
        self._prepare_prop(obs, detach_vlm=True)  # actor: VLM gradient blocked

        action = self._act_default(
            obs=obs,
            eval_mode=eval_mode,
            stddev=stddev,
            clip=None,
            use_target=False,
        )

        if unsqueezed:
            action = action.squeeze(0)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action

    def _act_default(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: float | None,
        use_target: bool,
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        dist = actor.forward(obs, stddev)

        # Only assert not training when this is called from the public act() method
        # (which is used for actual evaluation), not when called internally during training
        if eval_mode and not use_target:
            assert not self.training

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=clip)

        return action

    def update_critic(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        stddev: float,
        importance_weights: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            # use train mode as we use actor dropout
            assert self.actor_target.training

            # Predict next residual action and form the combined next action
            # Use target_action_noise config to control whether to add noise to target actions
            next_residual_action = self._act_default(
                obs=next_obs,
                eval_mode=not self.cfg.target_action_noise,  # Disable noise if target_action_noise=False
                stddev=stddev,
                clip=self.cfg.stddev_clip,
                use_target=True,
            )

            if self.residual_actor:
                # Current step: 'action' from the replay buffer is the executed combined action
                # Next step: combine and clamp to match environment execution
                next_action = torch.clamp(next_obs["observation.base_action"] + next_residual_action, -1.0, 1.0)
            else:
                next_action = next_residual_action

            # Compute target Q using min over a random subset of 2 heads
            target_all = self.critic_target.q_value(next_obs["feat"], next_obs["observation.state"], next_action)
            target_q_min = target_all.squeeze(-1)  # [B]
            target_q = (reward + (discount * target_q_min)).detach()

        if self.cfg.clip_q_target_to_reward_range:
            # Clip Q-target to prevent overestimation
            # For rewards in [0,1]: Q should be in [0, ~10] with gamma=0.99 and typical episode length
            target_q = torch.clamp(target_q, min=0, max=10.0)

        td_errors = None

        if self.critic.loss_cfg.type == "hl_gauss":
            # Compute logits for current Q heads and average HL-Gauss loss across heads
            q_per_head, logits_per_head = self.critic(obs["feat"], obs["observation.state"], action, return_logits=True)
            K = logits_per_head.shape[0]
            losses = [self.critic.hl_loss(logits_per_head[i], target_q) for i in range(K)]
            critic_loss = torch.stack(losses).mean()
        elif self.critic.loss_cfg.type == "c51":
            # Compute logits for current Q heads and C51 distributional loss
            q_per_head, logits_per_head = self.critic(obs["feat"], obs["observation.state"], action, return_logits=True)

            # Get next state distribution for C51 target computation
            with torch.no_grad():
                _, next_logits = self.critic_target(
                    next_obs["feat"], next_obs["observation.state"], next_action, return_logits=True
                )
                # Take min over random subset of heads for next distribution (configurable via min_q_heads)
                num_heads = min(self.critic.cfg.min_q_heads, next_logits.shape[0])
                idx = torch.randperm(next_logits.shape[0], device=next_logits.device)[:num_heads]
                next_logits_min = torch.min(next_logits.index_select(0, idx), dim=0).values
                next_distribution = torch.softmax(next_logits_min, dim=-1)

                # Project the target distribution
                # The discount factor passed to this function is already discount = gamma * (1 - done)
                # For C51, we need to extract the done mask and gamma separately
                # We'll use a simple heuristic: if discount is 0, then done=1, otherwise done=0
                dones = (discount == 0.0).float()
                gamma = 0.99  # Assume standard gamma value
                target_distribution = self.critic.c51_loss.project_distribution(next_distribution, reward, dones, gamma)

            # Compute C51 loss for each head
            K = logits_per_head.shape[0]
            losses = [self.critic.c51_loss(logits_per_head[i], target_distribution) for i in range(K)]
            critic_loss = torch.stack(losses).mean()
        else:
            q_all = self.critic(obs["feat"], obs["observation.state"], action).squeeze(-1)  # [K,B]
            # Compute TD errors for prioritized experience replay (before taking mean)
            td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)  # [B] - mean across heads

            # Apply importance sampling weights if provided (for prioritized experience replay)
            if importance_weights is not None:
                # Weight the squared TD errors by importance sampling weights
                weighted_td_errors = td_errors**2 * importance_weights
                critic_loss = weighted_td_errors.mean()
            else:
                # Mean squared error across heads and batch (uniform sampling)
                critic_loss = (td_errors**2).mean()

        metrics = {}
        metrics["train/critic_qt"] = target_q.mean().item()
        metrics["train/critic_loss"] = critic_loss.item()

        # ── Critic health checks ──
        assert not torch.isnan(critic_loss), "FATAL: critic_loss is NaN"
        assert not torch.isinf(critic_loss), "FATAL: critic_loss is Inf"
        assert not torch.isnan(target_q).any(), "FATAL: target_q contains NaN"
        assert critic_loss.item() < 1e6, f"FATAL: critic_loss exploded: {critic_loss.item():.2f}"
        qt_absmax = target_q.abs().max().item()
        assert qt_absmax < 100.0, (
            f"FATAL: target_q out of range: [{target_q.min().item():.2f}, {target_q.max().item():.2f}]. "
            f"Q-target clipping may not be working."
        )
        # Track Q-value statistics for monitoring
        metrics["train/critic_qt_min"] = target_q.min().item()
        metrics["train/critic_qt_max"] = target_q.max().item()
        metrics["train/critic_qt_std"] = target_q.std().item()

        # Store target_q for potential logging (calculated only when needed)
        metrics["_target_q"] = target_q.detach().cpu()
        # Store TD errors for prioritized experience replay
        if td_errors is not None:
            metrics["_td_errors"] = td_errors.detach().cpu()
        # Log importance sampling weights for monitoring PER behavior
        if importance_weights is not None:
            metrics["train/importance_weights_mean"] = importance_weights.mean().item()
            metrics["train/importance_weights_std"] = importance_weights.std().item()
            metrics["train/importance_weights_min"] = importance_weights.min().item()
            metrics["train/importance_weights_max"] = importance_weights.max().item()

        # Zero gradients
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward(retain_graph=True)

        # Gradient clipping
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoders.parameters(), self.cfg.critic_grad_clip_norm)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip_norm)

        # Store gradient norms for logging
        metrics["train/encoder_grad_norm"] = encoder_grad_norm.item()
        metrics["train/critic_grad_norm"] = critic_grad_norm.item()

        # ── Critic/Encoder gradient health checks ──
        assert not torch.isnan(encoder_grad_norm), "FATAL: encoder grad norm is NaN"
        assert not torch.isnan(critic_grad_norm), "FATAL: critic grad norm is NaN"
        assert encoder_grad_norm.item() < 1e4, f"FATAL: encoder grad norm too large: {encoder_grad_norm.item():.2f}"
        assert critic_grad_norm.item() < 1e4, f"FATAL: critic grad norm too large: {critic_grad_norm.item():.2f}"

        self.encoder_opt.step()
        self.critic_opt.step()

        return metrics

    def _compute_actor_loss(self, obs: dict[str, torch.Tensor], stddev: float):
        assert "feat" in obs, "safety check"

        action_pred: torch.Tensor = self._act_default(
            obs=obs,
            eval_mode=False,
            # stddev=stddev,
            # NOTE: This fix has not been fully verified yet.
            stddev=0.0,
            clip=self.cfg.stddev_clip,
            use_target=False,
        )

        # Add L2 regularization on action magnitude (before we add the residual action to the base)
        action_l2_penalty = self.cfg.actor.action_l2_reg_weight * torch.mean(torch.sum(action_pred**2, dim=-1))

        if self.residual_actor:
            # Create the full action by combining the base action and the residual action
            # and clamp to the valid range [-1, 1] to match environment execution
            combined_action = torch.clamp(obs["observation.base_action"] + action_pred, -1.0, 1.0)
        else:
            combined_action = action_pred

        q = self.critic.q_value_for_policy(obs["feat"], obs["observation.state"], combined_action)
        actor_loss_base = -q.mean()

        actor_loss_total = actor_loss_base + action_l2_penalty

        return actor_loss_total, actor_loss_base, combined_action, action_pred, action_l2_penalty

    def _compute_actor_bc_loss(self, batch, *, backprop_encoder):
        assert not self.residual_actor, "Not implemented"
        obs: dict[str, torch.Tensor] = batch["obs"]

        assert "feat" not in obs, "safety check"
        obs["feat"] = self._encode(obs, augment=True)

        if not backprop_encoder:
            obs["feat"] = obs["feat"].detach()

        pred_action = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=0,
            clip=None,
            use_target=False,
        )
        action: torch.Tensor = batch["action"]
        loss = nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss  # noqa: RET504

    def update_actor(self, obs: dict[str, torch.Tensor], stddev: float):
        metrics = {}

        # Compute actor loss and get the actions used (single actor call)
        (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
        ) = self._compute_actor_loss(obs, stddev)

        metrics["train/actor_loss_base"] = actor_loss_base.item()
        metrics["train/actor_loss_total"] = actor_loss_total.item()
        metrics["_actions"] = action_pred.detach().cpu()
        metrics["_combined_actions"] = combined_action.detach().cpu()

        # ── Actor health checks ──
        assert not torch.isnan(actor_loss_total), "FATAL: actor_loss is NaN"
        assert not torch.isinf(actor_loss_total), "FATAL: actor_loss is Inf"
        assert actor_loss_total.abs().item() < 1e6, f"FATAL: actor_loss exploded: {actor_loss_total.item():.2f}"
        assert not torch.isnan(action_pred).any(), "FATAL: actor output (residual) contains NaN"
        # Residual should be bounded by action_scale (tanh output * scale)
        residual_max = action_pred.abs().max().item()
        assert residual_max <= self.cfg.actor.action_scale + 1e-4, (
            f"FATAL: residual magnitude {residual_max:.4f} exceeds action_scale {self.cfg.actor.action_scale}"
        )
        metrics["train/residual_abs_max"] = residual_max
        metrics["train/residual_abs_mean"] = action_pred.abs().mean().item()

        # Log L2 regularization penalty if applied
        if self.cfg.actor.action_l2_reg_weight > 0:
            metrics["train/actor_l2_penalty"] = action_l2_penalty.item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_total.backward()

        # Gradient clipping
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip_norm)

        # Store gradient norm for logging
        metrics["train/actor_grad_norm"] = actor_grad_norm.item()

        # ── Actor gradient health checks ──
        assert not torch.isnan(actor_grad_norm), "FATAL: actor grad norm is NaN (exploded gradients)"
        assert actor_grad_norm.item() < 1e4, f"FATAL: actor grad norm too large: {actor_grad_norm.item():.2f}"

        self.actor_opt.step()

        # ── Post-step weight health check (sampled) ──
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                assert not torch.isnan(param).any(), f"FATAL: NaN in actor param '{name}' after optimizer step"
                break  # only check first param for speed

        return metrics

    def update_actor_rft(
        self,
        obs: dict[str, torch.Tensor],
        stddev: float,
        bc_batch,
        ref_agent: QAgent,
    ):
        metrics = {}

        # Compute actor loss and get the actions used (single actor call)
        (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
        ) = self._compute_actor_loss(obs, stddev)

        metrics["train/actor_loss_base"] = actor_loss_base.item()
        metrics["train/actor_loss_total"] = actor_loss_total.item()
        # Store residual actions for logging (the actual residual component we want to monitor)
        metrics["_actions"] = action_pred.detach().cpu()
        # Also store combined actions if needed for other purposes
        metrics["_combined_actions"] = combined_action.detach().cpu()

        # Log L2 regularization penalty if applied
        if self.cfg.actor.action_l2_reg_weight > 0:
            metrics["train/actor_l2_penalty"] = action_l2_penalty.item()

        # Use config option to control whether BC loss updates encoder
        bc_backprop_encoder = self.cfg.bc_backprop_encoder
        bc_loss = self._compute_actor_bc_loss(bc_batch, backprop_encoder=bc_backprop_encoder)
        assert actor_loss_total.size() == bc_loss.size()

        ratio = 1
        if self.cfg.bc_loss_dynamic:
            with torch.no_grad(), utils.eval_mode(self, ref_agent):
                assert ref_agent.cfg.act_method == "rl"

                # temporarily change to rl since we want to regularize actor not hybrid
                act_method = self.cfg.act_method
                self.cfg.act_method = "rl"

                ref_bc_obs = bc_batch.obs.copy()  # shallow copy
                ref_action = ref_agent.act(ref_bc_obs, eval_mode=True, cpu=False)

                # we first get the ref_action and then pop the feature
                # then we get the curr_action so that the obs["feat"] is the current feature
                # which can be used for computing q-values
                bc_obs = bc_batch.obs
                curr_action = self.act(bc_obs, eval_mode=True, cpu=False)

                curr_q = self.critic.q_value_for_policy(bc_obs["feat"], bc_obs["observation.state"], curr_action)
                ref_q = self.critic.q_value_for_policy(bc_obs["feat"], bc_obs["observation.state"], ref_action)

                ratio = (ref_q > curr_q).float().mean().item()

                # recover to original act_method
                self.cfg.act_method = act_method

        loss = actor_loss_total + (self.cfg.bc_loss_coef * ratio * bc_loss).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        # Conditionally update encoder along with actor if BC loss should backprop
        if bc_backprop_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)

        loss.backward()

        # Gradient clipping
        metrics["train/actor_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.actor_grad_clip_norm
        ).item()

        if bc_backprop_encoder:
            metrics["train/encoder_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                self.encoders.parameters(), self.cfg.actor_grad_clip_norm
            ).item()

        if bc_backprop_encoder:
            self.encoder_opt.step()
        self.actor_opt.step()

        metrics["rft/bc_loss"] = bc_loss.mean().item()
        metrics["rft/ratio"] = ratio
        return metrics

    def update(
        self,
        batch,
        stddev,
        update_actor,
        bc_batch=None,
        ref_agent: QAgent | None = None,
    ):
        obs: dict[str, torch.Tensor] = batch["obs"]
        action: torch.Tensor = batch["action"]
        reward: torch.Tensor = batch[("next", "reward")]
        discount: torch.Tensor = batch["gamma"]
        next_nonterminal: torch.Tensor = batch["nonterminal"]
        next_obs: dict[str, torch.Tensor] = batch[("next", "obs")]

        # ── Strict batch validation ──
        B = action.shape[0]
        assert action.dim() == 2 and action.shape[1] == 7, f"update: action shape {action.shape} != (B,7)"
        assert reward.dim() == 1 or (reward.dim() == 2 and reward.shape[-1] == 1), f"update: reward shape {reward.shape}"
        assert "observation.state" in obs, "update: missing observation.state in batch obs"
        assert "observation.base_action" in obs, "update: missing observation.base_action in batch obs"
        state = obs["observation.state"]
        assert state.dim() == 2 and state.shape[0] == B, f"update: state shape {state.shape} vs batch {B}"
        expected_state_dim = 10  # base state dim (before VLM concat)
        assert state.shape[1] == expected_state_dim, (
            f"update: state dim {state.shape[1]} != {expected_state_dim}. "
            f"State must be 10D (EEF3+quat4+grip2+contact_force1). "
            f"VLM concat happens later via _prepare_prop()."
        )
        ba = obs["observation.base_action"]
        assert ba.dim() == 2 and ba.shape == (B, 7), f"update: base_action shape {ba.shape} != ({B},7)"
        for cam in self.rl_cameras:
            assert cam in obs, f"update: missing camera '{cam}' in batch obs"
            img = obs[cam]
            expected_c = 1 if cam.startswith("observation.depth") else 3
            assert img.dim() == 4 and img.shape[0] == B and img.shape[1] == expected_c, (
                f"update: camera '{cam}' shape {img.shape} invalid (expected C={expected_c})"
            )
        # Check next_obs too
        assert "observation.state" in next_obs, "update: missing observation.state in batch next_obs"
        assert next_obs["observation.state"].shape == state.shape, (
            f"update: next_obs state shape {next_obs['observation.state'].shape} != obs state {state.shape}"
        )
        # Check for NaN/Inf
        assert not torch.isnan(action).any(), "update: NaN in action"
        assert not torch.isnan(reward).any(), "update: NaN in reward"
        assert not torch.isnan(state).any(), "update: NaN in state"
        assert not torch.isinf(reward).any(), "update: Inf in reward"

        # ── VLM latent batch validation ──
        if self.vlm_projector is not None:
            assert "observation.vlm_latent" in obs, (
                "update: VLM projector enabled but 'observation.vlm_latent' missing in batch obs. "
                "Check lowdim_keys and offline buffer."
            )
            vlm_batch = obs["observation.vlm_latent"]
            assert vlm_batch.dim() == 2 and vlm_batch.shape == (B, self.vlm_latent_dim), (
                f"update: vlm_latent shape {vlm_batch.shape} != expected ({B}, {self.vlm_latent_dim})"
            )
            assert not torch.isnan(vlm_batch).any(), "update: NaN in vlm_latent"
            # Check vlm_latent is not all zeros (at least some should be non-zero)
            vlm_nonzero = (vlm_batch.abs() > 1e-8).any(dim=-1).float().mean().item()
            assert vlm_nonzero > 0.1, (
                f"update: vlm_latent is mostly zeros ({vlm_nonzero*100:.0f}% non-zero rows). "
                f"VLM features not being extracted properly."
            )
            # Same for next_obs
            assert "observation.vlm_latent" in next_obs, "update: vlm_latent missing in next_obs"

        # To not bootstrap on terminal states we zero out the discount factor for terminal next states
        effective_discount = discount * next_nonterminal

        # ── Asymmetric critic validation (first 3 updates) ──
        if not hasattr(self, '_asym_debug_n'):
            self._asym_debug_n = 0
        if self.asymmetric_critic and self._asym_debug_n < 3:
            # Verify depth images are in the batch
            for cam in self.rl_cameras:
                assert cam in obs, f"ASYMMETRIC: missing '{cam}' in obs batch!"
                v = obs[cam]
                assert v.dim() == 4, f"ASYMMETRIC: {cam} must be 4D, got {v.shape}"
                assert (v.abs() > 1e-6).any(), f"ASYMMETRIC: {cam} is all zeros!"
                if cam in next_obs:
                    assert (next_obs[cam].abs() > 1e-6).any(), f"ASYMMETRIC: next_{cam} is all zeros!"
            # Verify object state
            if self.object_state_dim > 0:
                assert "observation.object_state" in obs, \
                    f"ASYMMETRIC: missing observation.object_state in batch! obj_dim={self.object_state_dim}"
                os_v = obs["observation.object_state"]
                assert os_v.shape[-1] == self.object_state_dim, \
                    f"ASYMMETRIC: object_state dim {os_v.shape[-1]} != expected {self.object_state_dim}"
                assert (os_v.abs() > 1e-8).any(), "ASYMMETRIC: object_state is all zeros!"
                assert "observation.object_state" in next_obs, "ASYMMETRIC: missing object_state in next_obs"
            print(f"[ASYMMETRIC DEBUG #{self._asym_debug_n}] "
                  f"obs.state={obs['observation.state'].shape} "
                  f"obj_state={'observation.object_state' in obs}({self.object_state_dim}D) "
                  f"depth_cams={[c for c in self.rl_cameras if c in obs]} "
                  f"actor.state_only={self.actor.state_only} "
                  f"critic.state_only={self.critic.q_ensemble.state_only if hasattr(self.critic, 'q_ensemble') else '?'}")
            self._asym_debug_n += 1

        obs["feat"] = self._encode(obs, augment=True)
        self._prepare_prop(obs, detach_vlm=False)  # critic: VLM gradient flows

        with torch.no_grad():
            next_obs["feat"] = self._encode(next_obs, augment=True)
            self._prepare_prop(next_obs, detach_vlm=False)  # next_obs for target Q

        # ── Post-encode asymmetric validation ──
        if self.asymmetric_critic and self._asym_debug_n <= 3:
            feat = obs["feat"]
            prop = obs["observation.state"]
            assert feat.shape[0] == B, f"ASYMMETRIC: feat batch {feat.shape[0]} != {B}"
            if feat.numel() > 0:
                assert (feat.abs() > 1e-8).any(), "ASYMMETRIC: encoded feat is all zeros!"
                print(f"[ASYMMETRIC DEBUG] post-encode: feat={feat.shape} "
                      f"feat_range=[{feat.min().item():.4f},{feat.max().item():.4f}] "
                      f"prop(after_prepare)={prop.shape} "
                      f"prop_range=[{prop.min().item():.4f},{prop.max().item():.4f}]")

        metrics = {}
        metrics["data/batch_R"] = reward.mean().item()

        # Depth batch debug (first 3 updates only)
        if not hasattr(self, '_update_depth_debug_n'):
            self._update_depth_debug_n = 0
        if self._update_depth_debug_n < 3:
            for cam in self.rl_cameras:
                if cam.startswith("observation.depth") and cam in obs:
                    v = obs[cam]
                    n_zeros = (v.abs() < 1e-6).all(dim=-1).all(dim=-1).all(dim=-1).sum().item()
                    print(f"[DEPTH UPDATE DEBUG] batch {cam}: shape={v.shape} dtype={v.dtype} "
                          f"range=[{v.min().item():.4f}, {v.max().item():.4f}] "
                          f"zero_samples={n_zeros}/{v.shape[0]} "
                          f"mean={v.mean().item():.4f}")
            self._update_depth_debug_n += 1

        # Extract importance sampling weights if available (for prioritized experience replay)
        importance_weights = batch.get("_weight", None)

        critic_metric = self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            discount=effective_discount,
            next_obs=next_obs,
            stddev=stddev,
            importance_weights=importance_weights,
        )
        utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)
        metrics.update(critic_metric)

        if not update_actor:
            return metrics

        # NOTE: actor loss does not backprop into the encoder or VLM projector
        obs["feat"] = obs["feat"].detach()
        obs["observation.state"] = obs["observation.state"].detach()  # detach VLM proj gradient for actor

        if bc_batch is None:
            actor_metric = self.update_actor(obs, stddev)
        else:
            assert ref_agent is not None
            actor_metric = self.update_actor_rft(obs, stddev, bc_batch, ref_agent)

        utils.soft_update_params(self.actor, self.actor_target, self.cfg.critic_target_tau)
        metrics.update(actor_metric)

        return metrics

    def step_lr_schedulers(self):
        """Step the learning rate schedulers for warmup."""
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
