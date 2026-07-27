#!/bin/bash
# Run LAMBDA on SafetyPointGoal1-v0 for N seeds.
# Usage: bash run_lambda.sh [N_SEEDS] [STEPS]
# Default: 3 seeds, 1M steps each.

N_SEEDS=${1:-3}
STEPS=${2:-1000000}
LAMBDA_DIR="$HOME/la-mbda"
ENV_NAME="lambda_env"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$LAMBDA_DIR"

export MUJOCO_GL=osmesa

for SEED in $(seq 0 $((N_SEEDS - 1))); do
    LOG_DIR="results/lambda_pointgoal1/seed${SEED}"
    echo "=== Running seed $SEED → $LOG_DIR ==="
    python experiments/train.py \
        --log_dir "$LOG_DIR" \
        --environment sgymn_SafetyPointGoal1-v0 \
        --total_training_steps "$STEPS" \
        --safety \
        --seed "$SEED" \
        2>&1 | tee "${LOG_DIR}_train.log"
done

echo "All seeds done. Results in $LAMBDA_DIR/results/"
