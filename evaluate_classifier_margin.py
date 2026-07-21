"""
Evaluate safety classifier discrimination quality for both models.

Shows that the ρ=1 (regularized) latent space gives the safety classifier
a larger, more confident separation between safe and unsafe observations —
even though both classifiers achieve 100% accuracy at threshold=0.

Usage:
    python evaluate_classifier_margin.py
    CHECKPOINT=/mnt/t7shield/safe_lewm_rho1.pt \\
    CHECKPOINT_NOREG=/mnt/t7shield/safe_lewm_rho0.pt \\
    CLF_PATH=/mnt/t7shield/classifier.pt \\
    CLF_NOREG_PATH=/mnt/t7shield/classifier_rho0.pt \\
    DATA_PATH=/mnt/t7shield/safety_point_goal.pkl \\
    python evaluate_classifier_margin.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.classifier import ObstacleMLP
from safe_lewm.dataset import load_trajectories

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT       = os.environ.get("CHECKPOINT",       "/mnt/t7shield/safe_lewm_rho1.pt")
CHECKPOINT_NOREG = os.environ.get("CHECKPOINT_NOREG", "/mnt/t7shield/safe_lewm_rho0.pt")
CLF_PATH         = os.environ.get("CLF_PATH",         "/mnt/t7shield/classifier.pt")
CLF_NOREG_PATH   = os.environ.get("CLF_NOREG_PATH",   "/mnt/t7shield/classifier_rho0.pt")
DATA_PATH        = os.environ.get("DATA_PATH",        "/mnt/t7shield/safety_point_goal.pkl")
OUT_DIR          = os.environ.get("OUT_DIR",          "eval_wm_viz")
MAX_OBS          = int(os.environ.get("MAX_OBS",      "20000"))
BATCH            = int(os.environ.get("BATCH",        "256"))
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

Path(OUT_DIR).mkdir(exist_ok=True)
print(f"Device: {DEVICE}")


def load_encoder(ckpt_path):
    cfg = Config()
    m = SafeJEPA(cfg).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.encoder, cfg


def load_clf(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    c = ObstacleMLP(z_dim=ckpt["z_dim"], hidden_dim=ckpt["hidden_dim"],
                    depth=ckpt["depth"], dropout=ckpt.get("dropout", 0.1)).to(DEVICE)
    c.load_state_dict(ckpt["classifier_state"])
    c.eval()
    zm = ckpt["z_mean"].to(DEVICE)
    zs = ckpt["z_std"].to(DEVICE)
    return c, zm, zs, ckpt["safe_threshold"]


print("Loading models and classifiers...")
enc_reg,   cfg = load_encoder(CHECKPOINT)
enc_noreg, _   = load_encoder(CHECKPOINT_NOREG)
clf_reg,   zm_reg,   zs_reg,   thr_reg   = load_clf(CLF_PATH)
clf_noreg, zm_noreg, zs_noreg, thr_noreg = load_clf(CLF_NOREG_PATH)

# ── Load observations ─────────────────────────────────────────────────────────
print("\nLoading trajectories...")
trajs = load_trajectories(DATA_PATH)

obs_list, cost_list = [], []
for traj in trajs:
    for t in range(len(traj["obs"])):
        obs_list.append(traj["obs"][t])
        cost_list.append(traj["cost"][t])
    if len(obs_list) >= MAX_OBS:
        break

obs_arr  = np.array(obs_list[:MAX_OBS], dtype=np.float32)
cost_arr = np.array(cost_list[:MAX_OBS], dtype=np.float32)
labels   = cost_arr.astype(int)  # 0=safe, 1=unsafe

n_safe   = int((labels == 0).sum())
n_unsafe = int((labels == 1).sum())
print(f"Total: {len(obs_arr)} obs | safe={n_safe} | unsafe={n_unsafe}")

# Use held-out 15% (same split seed as train_classifier.py)
rng = np.random.default_rng(42)
safe_idx   = np.where(labels == 0)[0]
unsafe_idx = np.where(labels == 1)[0]

def split_idx(idx, rng, train_frac=0.70, cal_frac=0.15):
    idx = rng.permutation(idx)
    n = len(idx)
    n_train = int(n * train_frac)
    n_cal   = int(n * cal_frac)
    return idx[:n_train], idx[n_train:n_train+n_cal], idx[n_train+n_cal:]

_, _, safe_te   = split_idx(safe_idx, rng)
_, _, unsafe_te = split_idx(unsafe_idx, rng)
te_idx = np.concatenate([safe_te, unsafe_te])
te_labels = labels[te_idx]

print(f"Test set: {len(te_idx)} obs (safe={len(safe_te)}, unsafe={len(unsafe_te)})")

# ── Encode and score ───────────────────────────────────────────────────────────
def encode_and_score(encoder, clf, zm, zs, obs_arr, idx):
    scores = []
    for i in range(0, len(idx), BATCH):
        batch_idx = idx[i:i+BATCH]
        x = torch.tensor(obs_arr[batch_idx]).to(DEVICE)
        with torch.no_grad():
            z = encoder(x)
            z_norm = (z - zm) / zs
            s = clf(z_norm).cpu().numpy()
        scores.append(s)
    return np.concatenate(scores)


print("\nScoring test set with ρ=1 classifier...")
scores_reg = encode_and_score(enc_reg, clf_reg, zm_reg, zs_reg, obs_arr, te_idx)

print("Scoring test set with ρ=0 classifier...")
scores_noreg = encode_and_score(enc_noreg, clf_noreg, zm_noreg, zs_noreg, obs_arr, te_idx)

safe_mask   = te_labels == 0
unsafe_mask = te_labels == 1

# ── Print statistics ───────────────────────────────────────────────────────────
def stats(scores, mask, name):
    s = scores[mask]
    return f"{name}: mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}"

print("\n── Score statistics ───────────────────────────────────────────────────────")
print("ρ=1 (regularized):")
print("  " + stats(scores_reg, safe_mask,   "  Safe  "))
print("  " + stats(scores_reg, unsafe_mask, "  Unsafe"))
margin_reg = scores_reg[safe_mask].mean() - scores_reg[unsafe_mask].mean()
print(f"  Margin (safe_mean - unsafe_mean): {margin_reg:.4f}")

print("ρ=0 (no regularization):")
print("  " + stats(scores_noreg, safe_mask,   "  Safe  "))
print("  " + stats(scores_noreg, unsafe_mask, "  Unsafe"))
margin_noreg = scores_noreg[safe_mask].mean() - scores_noreg[unsafe_mask].mean()
print(f"  Margin (safe_mean - unsafe_mean): {margin_noreg:.4f}")

print(f"\nMargin improvement with ρ=1: {(margin_reg - margin_noreg) / abs(margin_noreg) * 100:+.1f}%")

# AUC (higher score = safe prediction, so flip sign for unsafe positive)
auc_reg   = roc_auc_score(te_labels, -scores_reg)    # unsafe=1 is positive class
auc_noreg = roc_auc_score(te_labels, -scores_noreg)
print(f"\nROC-AUC (detecting unsafe obs):")
print(f"  ρ=1: {auc_reg:.4f}")
print(f"  ρ=0: {auc_noreg:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Safety Classifier Score Distribution: ρ=1 vs ρ=0", fontsize=13)

bins = np.linspace(-4, 4, 60)

for ax, scores, label, auc in [
    (axes[0], scores_reg,   "ρ=1 (with regularization)", auc_reg),
    (axes[1], scores_noreg, "ρ=0 (no regularization)",   auc_noreg),
]:
    ax.hist(scores[safe_mask],   bins=bins, alpha=0.6, color="steelblue", label=f"Safe (n={safe_mask.sum()})",   density=True)
    ax.hist(scores[unsafe_mask], bins=bins, alpha=0.7, color="tomato",    label=f"Unsafe (n={unsafe_mask.sum()})", density=True)
    ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Threshold=0")
    ax.set_title(f"{label}\nAUC={auc:.4f}")
    ax.set_xlabel("Classifier score  (+ = safe, − = unsafe)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = f"{OUT_DIR}/classifier_score_distribution.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")
