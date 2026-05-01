#!/bin/bash
source /home/nas_main/kinamkim/.venvs/groot/bin/activate
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
python scripts/debug_axes.py
