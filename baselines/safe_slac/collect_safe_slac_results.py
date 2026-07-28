"""
Parse safe-slac tensorboard logs and print mean ± std over seeds.
Usage: python baselines/safe_slac/collect_safe_slac_results.py --log_dir ~/safe-slac/results/pointgoal1
"""
import argparse
import numpy as np
from pathlib import Path

try:
    from tensorflow.core.util import event_pb2
    from tensorflow.python.lib.io import tf_record
    USE_TF = True
except ImportError:
    USE_TF = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    USE_TB = True
except ImportError:
    USE_TB = False


def load_scalar(log_dir, tag):
    if USE_TB:
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        try:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            vals  = np.array([e.value for e in events])
            return steps, vals
        except KeyError:
            return None, None
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="~/safe-slac/results/pointgoal1")
    parser.add_argument("--last_n", type=int, default=10,
                        help="Average over last N eval points per seed")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser()
    seed_dirs = sorted(log_dir.glob("seed*"))
    if not seed_dirs:
        print(f"No seed dirs found in {log_dir}")
        return

    returns, costs = [], []
    for sd in seed_dirs:
        steps_r, vals_r = load_scalar(sd, "train/return")
        steps_c, vals_c = load_scalar(sd, "train/cost")
        if vals_r is None:
            # try alternate tag names
            steps_r, vals_r = load_scalar(sd, "return")
            steps_c, vals_c = load_scalar(sd, "cost")
        if vals_r is not None and len(vals_r) > 0:
            returns.append(float(np.mean(vals_r[-args.last_n:])))
        if vals_c is not None and len(vals_c) > 0:
            costs.append(float(np.mean(vals_c[-args.last_n:])))
        print(f"  {sd.name}: return={vals_r[-1] if vals_r is not None and len(vals_r)>0 else 'N/A':.2f}  "
              f"cost={vals_c[-1] if vals_c is not None and len(vals_c)>0 else 'N/A'}")

    print()
    if returns:
        print(f"Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    if costs:
        print(f"Cost:   {np.mean(costs):.2f} ± {np.std(costs):.2f}")


if __name__ == "__main__":
    main()
