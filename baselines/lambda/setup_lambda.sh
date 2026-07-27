#!/bin/bash
# Setup script for LAMBDA baseline on anderson-accelerator
# Run from any directory; clones la-mbda and creates conda env.

set -e

LAMBDA_DIR="$HOME/la-mbda"
ENV_NAME="lambda_env"

# ── 1. Clone repo ──────────────────────────────────────────────────────────────
if [ ! -d "$LAMBDA_DIR" ]; then
    git clone https://github.com/yardenas/la-mbda.git "$LAMBDA_DIR"
fi
cd "$LAMBDA_DIR"

# ── 2. Create conda env ────────────────────────────────────────────────────────
conda create -y -n "$ENV_NAME" python=3.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 3. Install MuJoCo 2.1 (free, no license needed) ──────────────────────────
pip install mujoco==2.3.7   # mujoco bindings without mujoco-py

# ── 4. Install core deps (upgraded from py3.6 to py3.8 compatible versions) ──
pip install \
    tensorflow==2.9.0 \
    tensorflow-probability==0.17.0 \
    tf-agents==0.13.0 \
    numpy==1.23.5 \
    absl-py \
    gin-config \
    tensorboard \
    dm-env \
    imageio \
    matplotlib

# ── 5. Install safety_gymnasium + gymnasium shim ──────────────────────────────
pip install safety-gymnasium
pip install shimmy[gym-v21]   # gymnasium<->gym compatibility shim

# ── 6. Patch env_wrappers.py to support safety_gymnasium ─────────────────────
python - <<'EOF'
import re, pathlib

wrappers = pathlib.Path("la_mbda/env_wrappers.py").read_text()

# Add safety_gymnasium import after existing imports
if "import safety_gymnasium" not in wrappers:
    wrappers = wrappers.replace(
        "import gym\n",
        "import gym\nimport safety_gymnasium\n"
    )

# Add sgymn_ branch for safety_gymnasium envs
new_branch = """
  elif name.startswith('sgymn'):
    import safety_gymnasium
    env_name = '_'.join(name.split('_')[1:])
    env = safety_gymnasium.make(env_name, render_mode='rgb_array')
    # Wrap to gym-compatible API (4-tuple step, cost in info)
    env = SafetyGymnasiumWrapper(env)
"""

if "sgymn" not in wrappers:
    wrappers = wrappers.replace(
        "  elif name.startswith('sgym'):",
        new_branch + "  elif name.startswith('sgym'):"
    )

pathlib.Path("la_mbda/env_wrappers.py").write_text(wrappers)
print("Patched env_wrappers.py")
EOF

# ── 7. Add SafetyGymnasiumWrapper class to env_wrappers.py ───────────────────
python - <<'EOF'
import pathlib

wrapper_class = '''

class SafetyGymnasiumWrapper:
    """Wraps safety_gymnasium env to gym-compatible 4-tuple step API."""
    def __init__(self, env):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        obs, info = self._env.reset()
        return obs

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        info['cost'] = cost
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped
'''

path = pathlib.Path("la_mbda/env_wrappers.py")
text = path.read_text()
if "SafetyGymnasiumWrapper" not in text:
    text = text + wrapper_class
    path.write_text(text)
    print("Added SafetyGymnasiumWrapper class")
EOF

# ── 8. Install la-mbda package ────────────────────────────────────────────────
pip install -e .

echo ""
echo "Setup complete. To run LAMBDA:"
echo "  conda activate $ENV_NAME"
echo "  cd $LAMBDA_DIR"
echo "  python experiments/train.py \\"
echo "    --log_dir results/lambda_pointgoal1/seed0 \\"
echo "    --environment sgymn_SafetyPointGoal1-v0 \\"
echo "    --total_training_steps 1000000 \\"
echo "    --safety"
