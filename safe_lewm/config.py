from dataclasses import dataclass, field


@dataclass
class Config:
    # Environment
    env_name: str = "SafetyPointGoal1-v0"
    image_size: int = 64
    frame_stack: int = 4
    frame_skip: int = 2

    # Data collection
    num_episodes: int = 5000
    episode_length: int = 200
    data_path: str = "data/safety_point_goal.pkl"

    # Model
    obs_channels: int = 12  # frame_stack * 3 RGB
    z_dim: int = 256
    action_dim: int = 2  # PointGoal action space
    hidden_dim: int = 256

    # Action embedder
    action_smoothed_dim: int = 32
    action_emb_dim: int = 64

    # CNN encoder
    cnn_channels: list = field(default_factory=lambda: [32, 64, 128, 256])

    # ARPredictor
    pred_depth: int = 4
    pred_heads: int = 4
    pred_dim_head: int = 64
    pred_mlp_dim: int = 512
    num_frames: int = 5  # trajectory length T

    # Transition model
    trans_hidden: int = 256

    # SIGReg
    sig_knots: int = 17
    sig_num_proj: int = 1024

    # Loss weights
    lambda_sig: float = 0.1
    lambda_bsim: float = 1.0
    lambda_cost: float = 1.0

    # Bisimulation
    rho: float = 0.1      # cost weight in bisimulation metric
    gamma: float = 0.99   # discount

    # Training
    batch_size: int = 64
    seq_len: int = 5       # T sub-trajectory length
    lr: float = 1e-4
    train_steps: int = 200000
    log_every: int = 1000
    save_every: int = 10000
    checkpoint_path: str = "checkpoints/safe_lewm.pt"

    # CEM planner
    cem_samples: int = 200
    cem_iterations: int = 20
    cem_elites: int = 20
    cem_horizon: int = 5
    alpha_safety: float = 10.0
    safety_threshold: float = 0.5
    history_size: int = 3

    # Evaluation
    eval_episodes: int = 50
