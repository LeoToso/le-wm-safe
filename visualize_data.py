"""Visualize the collected Safety Gym dataset."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_PATH = "data/safety_point_goal.pkl"

with open(DATA_PATH, "rb") as f:
    trajs = pickle.load(f)

print(f"Loaded {len(trajs)} trajectories")
print(f"Obs shape per traj: {trajs[0]['obs'].shape}")   # (T, 12, 64, 64)
print(f"Action dim: {trajs[0]['action'].shape[-1]}")

# --- Aggregate stats ---
ep_rewards = [t["reward"].sum() for t in trajs]
ep_costs   = [t["cost"].sum() for t in trajs]
ep_lengths = [len(t["reward"]) for t in trajs]
cost_rate  = [t["cost"].mean() for t in trajs]

print(f"\nReward  — mean: {np.mean(ep_rewards):.2f}, std: {np.std(ep_rewards):.2f}")
print(f"Cost    — mean: {np.mean(ep_costs):.2f},  std: {np.std(ep_costs):.2f}")
print(f"Cost rate per step — mean: {np.mean(cost_rate):.3f}")
print(f"Episodes with any cost: {sum(c > 0 for c in ep_costs)}/{len(trajs)}")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

# 1. Episode reward distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(ep_rewards, bins=40, color="#4C9BE8", edgecolor="white", linewidth=0.5)
ax1.axvline(np.mean(ep_rewards), color="red", linestyle="--", label=f"mean={np.mean(ep_rewards):.2f}")
ax1.set_title("Episode Reward Distribution"); ax1.set_xlabel("Total Reward"); ax1.legend()

# 2. Episode cost distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(ep_costs, bins=40, color="#E8844C", edgecolor="white", linewidth=0.5)
ax2.axvline(np.mean(ep_costs), color="red", linestyle="--", label=f"mean={np.mean(ep_costs):.2f}")
ax2.set_title("Episode Cost Distribution"); ax2.set_xlabel("Total Cost"); ax2.legend()

# 3. Reward vs Cost scatter
ax3 = fig.add_subplot(gs[0, 2])
sc = ax3.scatter(ep_rewards, ep_costs, alpha=0.4, s=8, c=ep_costs, cmap="YlOrRd")
plt.colorbar(sc, ax=ax3, label="Cost")
ax3.set_title("Reward vs Cost per Episode")
ax3.set_xlabel("Total Reward"); ax3.set_ylabel("Total Cost")

# 4. Sample frames from a trajectory (first 4 frames of traj 0, channel 0 = first RGB frame)
ax4 = fig.add_subplot(gs[1, 0])
traj = trajs[0]
# obs shape: (T, 12, 64, 64) — take channels 0,1,2 (first stacked frame, RGB)
frame = traj["obs"][0, :3]  # (3, 64, 64)
# Denormalize from ImageNet stats
mean = np.array([0.485, 0.456, 0.406])[:, None, None]
std  = np.array([0.229, 0.224, 0.225])[:, None, None]
frame = (frame * std + mean).clip(0, 1).transpose(1, 2, 0)
ax4.imshow(frame)
ax4.set_title("Sample Frame (traj 0, step 0)"); ax4.axis("off")

# 5. Cost signal over a trajectory
ax5 = fig.add_subplot(gs[1, 1])
# pick a trajectory with some costs
costly = sorted(trajs, key=lambda t: t["cost"].sum(), reverse=True)[0]
ax5.fill_between(range(len(costly["cost"])), costly["cost"], alpha=0.6, color="#E8844C", label="cost")
ax5.plot(costly["reward"], color="#4C9BE8", linewidth=0.8, label="reward")
ax5.set_title("Cost & Reward Signal (highest-cost traj)")
ax5.set_xlabel("Step"); ax5.legend()

# 6. Action distribution
ax6 = fig.add_subplot(gs[1, 2])
all_actions = np.concatenate([t["action"] for t in trajs], axis=0)
ax6.hist(all_actions[:, 0], bins=50, alpha=0.6, label="action[0]", color="#4C9BE8")
ax6.hist(all_actions[:, 1], bins=50, alpha=0.6, label="action[1]", color="#E8844C")
ax6.set_title("Action Distribution (all steps)")
ax6.set_xlabel("Action value"); ax6.legend()

plt.suptitle(f"SafetyPointGoal1 Dataset — {len(trajs)} episodes", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("dataset_viz.png", dpi=150, bbox_inches="tight")
print("\nSaved dataset_viz.png")
