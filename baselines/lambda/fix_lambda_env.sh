#!/bin/bash
# Fix lambda_env after the failed setup.py install.
# Run from ~/la-mbda with lambda_env activated.
# Usage: conda activate lambda_env && bash ~/le-wm-safe/baselines/lambda/fix_lambda_env.sh

set -e
cd ~/la-mbda

echo "=== Step 1: Pin gymnasium back to 0.28.1 (required by safety_gymnasium) ==="
pip install "gymnasium==0.28.1" --force-reinstall -q

echo "=== Step 2: Install safety_gymnasium and its exact mujoco ==="
pip install "mujoco==2.3.3" -q
pip install safety-gymnasium -q

echo "=== Step 3: Install la-mbda deps we actually need (skip pinned setup.py) ==="
pip install \
    tensorflow==2.9.0 \
    tensorflow-probability==0.17.0 \
    "tf-agents>=0.13,<0.14" \
    "dm-control>=1.0" \
    "dm-env>=1.6" \
    gin-config \
    tensorboard \
    numpy==1.23.5 \
    absl-py \
    imageio \
    matplotlib \
    -q

echo "=== Step 4: Install la-mbda WITHOUT its setup.py deps ==="
pip install --no-deps -e . -q

echo "=== Step 5: Patch env_wrappers.py (idempotent) ==="
python - <<'EOF'
import pathlib

path = pathlib.Path("la_mbda/env_wrappers.py")
text = path.read_text()

# Ensure safety_gymnasium import is present
if "import safety_gymnasium" not in text:
    text = "import safety_gymnasium\n" + text

# Add SafetyGymnasiumWrapper class if missing
wrapper_class = '''

class SafetyGymnasiumWrapper:
    """Adapts safety_gymnasium (gymnasium API, 6-tuple step) to old gym 4-tuple API."""
    def __init__(self, env):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        result = self._env.reset()
        return result[0] if isinstance(result, tuple) else result

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        info = dict(info)
        info['cost'] = float(cost)
        done = bool(terminated or truncated)
        return obs, float(reward), done, info

    def render(self, *args, **kwargs):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped
'''

if "SafetyGymnasiumWrapper" not in text:
    text = text + wrapper_class

# Add sgymn_ branch if missing
sgymn_branch = """
  elif name.startswith('sgymn'):
    import safety_gymnasium as _sg
    env_name = '_'.join(name.split('_')[1:])
    raw = _sg.make(env_name, render_mode='rgb_array')
    return SafetyGymnasiumWrapper(raw)
"""
if "sgymn" not in text:
    text = text.replace(
        "  elif name.startswith('sgym'):",
        sgymn_branch + "  elif name.startswith('sgym'):"
    )

path.write_text(text)
print("env_wrappers.py patched OK")
EOF

echo ""
echo "=== Step 6: Smoke test ==="
MUJOCO_GL=osmesa python - <<'EOF'
import sys
sys.path.insert(0, '.')
from la_mbda.env_wrappers import make_env
env = make_env('sgymn_SafetyPointGoal1-v0', action_repeat=2, observation_type='state')
obs = env.reset()
print(f"reset OK — obs shape: {obs.shape}")
obs2, r, done, info = env.step(env.action_space.sample())
print(f"step OK  — reward={r:.3f}  cost={info.get('cost', 'N/A')}  done={done}")
env.close()
print("Smoke test PASSED")
EOF
