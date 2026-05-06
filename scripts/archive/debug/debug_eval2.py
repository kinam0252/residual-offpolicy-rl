#!/usr/bin/env python3
import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl")

from resfit.rl_finetuning.config.residual_td3_mujoco_drawer import ResidualTD3MuJoCoDrawerConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

cfg = ResidualTD3MuJoCoDrawerConfig()
cfg.agent.actor.action_scale = 0.1
cfg.agent.actor.hidden_dim = 512
cfg.agent.critic.hidden_dim = 1024
cfg.agent.device = "cpu"

agent = QAgent(
    obs_shape=(3, 84, 84),
    prop_shape=(8,),
    action_dim=7,
    rl_cameras=[],
    cfg=cfg.agent,
    residual_actor=True,
    object_state_dim=5,
    asymmetric_critic=False,
)

ckpt = torch.load('outputs/drawer_td3_v5_20260501_192236/checkpoints/best.pt', map_location='cpu', weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())}")
model_sd = ckpt["model"]

agent_keys = set(agent.state_dict().keys())
ckpt_keys = set(model_sd.keys())
missing = agent_keys - ckpt_keys
unexpected = ckpt_keys - agent_keys
print(f"Agent keys: {len(agent_keys)}, Ckpt keys: {len(ckpt_keys)}")
if missing: print(f"MISSING: {sorted(missing)[:10]}")
if unexpected: print(f"UNEXPECTED: {sorted(unexpected)[:10]}")
if not missing and not unexpected: print("Keys MATCH perfectly")

agent.load_state_dict(model_sd)
agent.eval()

# Test with typical obs values - mimicking what the env provides
# D2 test
fake_obs = {
    "observation.state": torch.tensor([[0.35, 0.0, 0.25, 0.0, 0.0, 0.7, 0.7, 0.0]]),
    "observation.base_action": torch.tensor([[0.1, -0.05, 0.02, 0.0, 0.0, 0.0, 0.0]]),
    "observation.object_state": torch.tensor([[0.0, 2.0, 0.35, 0.0, 0.22]]),
}
with torch.no_grad():
    action = agent.act(fake_obs, eval_mode=True, stddev=0.0, cpu=True)
print(f"\nD2 action (typical obs): {action.numpy().round(5)}")
print(f"  abs mean: {action.abs().mean().item():.6f}")

# D3 test
fake_obs["observation.object_state"] = torch.tensor([[0.0, 3.0, 0.35, 0.0, 0.28]])
with torch.no_grad():
    action = agent.act(fake_obs, eval_mode=True, stddev=0.0, cpu=True)
print(f"D3 action (typical obs): {action.numpy().round(5)}")
print(f"  abs mean: {action.abs().mean().item():.6f}")

# D4 test  
fake_obs["observation.object_state"] = torch.tensor([[0.0, 4.0, 0.35, 0.0, 0.35]])
with torch.no_grad():
    action = agent.act(fake_obs, eval_mode=True, stddev=0.0, cpu=True)
print(f"D4 action (typical obs): {action.numpy().round(5)}")
print(f"  abs mean: {action.abs().mean().item():.6f}")

# Check: what does zero residual look like (fresh agent)?
agent2 = QAgent(
    obs_shape=(3, 84, 84),
    prop_shape=(8,),
    action_dim=7,
    rl_cameras=[],
    cfg=cfg.agent,
    residual_actor=True,
    object_state_dim=5,
    asymmetric_critic=False,
)
agent2.eval()
fake_obs["observation.object_state"] = torch.tensor([[0.0, 2.0, 0.35, 0.0, 0.22]])
with torch.no_grad():
    action = agent2.act(fake_obs, eval_mode=True, stddev=0.0, cpu=True)
print(f"\nFresh agent (step 0) D2 action: {action.numpy().round(5)}")
print(f"  abs mean: {action.abs().mean().item():.6f}")
