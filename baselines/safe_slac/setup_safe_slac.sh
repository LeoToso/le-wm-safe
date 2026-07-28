#!/bin/bash
# Set up safe-slac baseline environment on the cluster
# Run from: bash baselines/safe_slac/setup_safe_slac.sh

set -e

cd ~

# Create conda env
conda create -n safe_slac_env python=3.8 -y
conda activate safe_slac_env || source activate safe_slac_env

PYTHON=/home/dz2478/miniconda3/envs/safe_slac_env/bin/python
PIP=/home/dz2478/miniconda3/envs/safe_slac_env/bin/pip

# Clone repo
if [ ! -d ~/safe-slac ]; then
    git clone https://github.com/safe-slac/safe-slac.git ~/safe-slac
fi
cd ~/safe-slac

# Core deps (pin torch to a version that works with CUDA 12.4 on Python 3.8)
$PIP install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
$PIP install gym==0.21.0  # 0.15.7 is too old; 0.21 still has old step API
$PIP install numpy==1.22.4 matplotlib==3.5.2 moviepy==1.0.3 pandas==1.1.2 \
             Pillow==9.1.1 tensorboard tensorboardX tqdm GitPython

# safety-gym (old OpenAI version)
$PIP install mujoco-py==2.1.2.14  # works without license for MuJoCo 2.1
$PIP install git+https://github.com/openai/safety-gym.git

echo "Done. Test with:"
echo "  MUJOCO_GL=egl /home/dz2478/miniconda3/envs/safe_slac_env/bin/python -c 'import safety_gym; import torch; print(torch.__version__)'"
