#!/bin/bash
# Run safe-slac on SafetyPointGoal1 for 3 seeds
# Usage: bash baselines/safe_slac/run_safe_slac.sh

PYTHON=/home/dz2478/miniconda3/envs/safe_slac_env/bin/python
LOGBASE=~/safe-slac/results/pointgoal1

mkdir -p $LOGBASE/seed{0,1,2}

for seed in 0 1 2; do
    nohup env MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 $PYTHON ~/safe-slac/train_safe.py \
        --domain_name Safexp \
        --task_name PointGoal1 \
        --seed $seed \
        --num_steps 100000 \
        --cuda \
        > $LOGBASE/seed${seed}/train.log 2>&1 &
    echo "Seed $seed PID: $!"
    sleep 5
done
