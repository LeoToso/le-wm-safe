import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from .config import Config
from .model import SafeJEPA
from .env_utils import SafetyGymWrapper
from .planner import CEMPlanner


def evaluate(cfg: Config, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SafeJEPA(cfg).to(device)
    ckpt = checkpoint_path or cfg.checkpoint_path
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    planner = CEMPlanner(model, cfg)
    env = SafetyGymWrapper(cfg.env_name, cfg.image_size, cfg.frame_stack, cfg.frame_skip)

    episode_rewards = []
    episode_costs = []

    for ep in range(cfg.eval_episodes):
        obs = env.reset()
        z_hist = deque(maxlen=cfg.history_size)
        ep_reward = 0.0
        ep_cost = 0.0
        done = False
        step = 0

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 12, 64, 64)
        with torch.no_grad():
            z0 = model.encoder(obs_t)  # (1, D)
        for _ in range(cfg.history_size):
            z_hist.append(z0)

        goal_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            z_goal = model.encoder(goal_obs)  # (1, D)

        while not done and step < cfg.episode_length:
            z_hist_tensor = torch.stack(list(z_hist), dim=1)  # (1, HS, D)

            with torch.no_grad():
                action = planner.plan(z_hist_tensor, z_goal, cfg.action_dim)

            next_obs, reward, cost, done, info = env.step(action)
            ep_reward += reward
            ep_cost += cost

            next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                z_next = model.encoder(next_obs_t)
            z_hist.append(z_next)

            step += 1

        episode_rewards.append(ep_reward)
        episode_costs.append(ep_cost)
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, cost={ep_cost:.2f}")

    avg_reward = np.mean(episode_rewards)
    avg_cost = np.mean(episode_costs)
    cost_rate = np.mean([c > 0 for c in episode_costs])

    print(f"\n=== Evaluation Results ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Cost:   {avg_cost:.2f}")
    print(f"Cost Rate:      {cost_rate:.2%}")

    # Random baseline
    print("\n=== Random Baseline ===")
    rand_rewards, rand_costs = [], []
    for ep in range(20):
        obs = env.reset()
        ep_r, ep_c = 0.0, 0.0
        for _ in range(cfg.episode_length):
            action = env.action_space.sample()
            _, r, c, done, _ = env.step(action)
            ep_r += r
            ep_c += c
            if done:
                break
        rand_rewards.append(ep_r)
        rand_costs.append(ep_c)
    print(f"Random Avg Reward: {np.mean(rand_rewards):.2f}")
    print(f"Random Avg Cost:   {np.mean(rand_costs):.2f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(episode_rewards, label="Safe LeWM", color="blue")
    ax1.axhline(np.mean(rand_rewards), color="red", linestyle="--", label="Random baseline")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward Return")
    ax1.set_title("Reward Return over Evaluation Episodes")
    ax1.legend()

    ax2.plot(episode_costs, label="Safe LeWM", color="orange")
    ax2.axhline(np.mean(rand_costs), color="red", linestyle="--", label="Random baseline")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cost Return")
    ax2.set_title("Cost Return over Evaluation Episodes")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("eval_results.png", dpi=150)
    plt.show()
    print("Saved plot to eval_results.png")

    return {
        "avg_reward": avg_reward,
        "avg_cost": avg_cost,
        "cost_rate": cost_rate,
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
    }


if __name__ == "__main__":
    cfg = Config()
    evaluate(cfg)
