"""
Parse LAMBDA tensorboard logs and print summary metrics
matching what we report for our LeWM models:
  - Total cost (collisions) over evaluation episodes
  - Average episodic reward
  - Average episodic cost

Usage:
    python collect_lambda_results.py --log_dir ~/la-mbda/results/lambda_pointgoal1
"""
import os, argparse
import numpy as np

def read_tb_scalars(log_dir, tag):
    """Read a scalar tag from all tfevents files under log_dir."""
    from tensorflow.core.util import event_pb2
    from tensorflow.python.lib.io import tf_record
    values = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if "tfevents" not in f:
                continue
            path = os.path.join(root, f)
            try:
                for rec in tf_record.tf_record_iterator(path):
                    ev = event_pb2.Event.FromString(rec)
                    for v in ev.summary.value:
                        if v.tag == tag:
                            values.append((ev.step, v.simple_value))
            except Exception:
                pass
    return sorted(values, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=os.path.expanduser("~/la-mbda/results/lambda_pointgoal1"))
    args = parser.parse_args()

    seed_dirs = sorted([
        os.path.join(args.log_dir, d)
        for d in os.listdir(args.log_dir)
        if os.path.isdir(os.path.join(args.log_dir, d))
    ])

    if not seed_dirs:
        print(f"No seed directories found in {args.log_dir}")
        return

    rewards, costs = [], []
    for sd in seed_dirs:
        r = read_tb_scalars(sd, "eval/average_return")
        c = read_tb_scalars(sd, "eval/average_cost")
        if r: rewards.append(r[-1][1])   # final value
        if c: costs.append(c[-1][1])
        print(f"  {os.path.basename(sd)}: reward={r[-1][1]:.3f if r else 'N/A'}, cost={c[-1][1]:.3f if c else 'N/A'}")

    print(f"\n── LAMBDA on SafetyPointGoal1-v0 ────────────────────────")
    if rewards:
        print(f"  Avg reward  : {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    if costs:
        print(f"  Avg cost    : {np.mean(costs):.3f} ± {np.std(costs):.3f}")


if __name__ == "__main__":
    main()
