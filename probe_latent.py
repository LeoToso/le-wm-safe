"""
Probe the latent space of the trained SafeJEPA encoder.
Checks:
  1. No collapse (latent vectors are diverse)
  2. Safe vs unsafe separation via PCA and t-SNE
  3. Linear separability: logistic regression on z predicts cost label
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.dataset import load_trajectories

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = "/mnt/t7shield/safe_lewm.pt"
DATA_PATH   = "/mnt/t7shield/safety_point_goal.pkl"
OUT_DIR     = Path("latent_probe")
MAX_SAMPLES = 5000   # how many obs to encode (safe + unsafe combined)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(exist_ok=True)

# Load model
print("Loading model...")
cfg = Config()
model = SafeJEPA(cfg).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# Load data
print("Loading trajectories...")
trajs = load_trajectories(DATA_PATH)

# Collect flat (obs, cost) pairs
print("Collecting observations...")
obs_list, cost_list = [], []
for traj in trajs:
    for t in range(len(traj["obs"])):
        obs_list.append(traj["obs"][t])    # (12, 64, 64)
        cost_list.append(traj["cost"][t])  # 0.0 or 1.0
    if len(obs_list) >= MAX_SAMPLES * 4:   # oversample then balance
        break

obs_arr  = np.array(obs_list,  dtype=np.float32)   # (N, 12, 64, 64)
cost_arr = np.array(cost_list, dtype=np.float32)   # (N,)

# Balance safe / unsafe so PCA isn't dominated by safe samples
safe_idx   = np.where(cost_arr == 0)[0]
unsafe_idx = np.where(cost_arr == 1)[0]
print(f"Total: {len(obs_arr)} obs | safe={len(safe_idx)} | unsafe={len(unsafe_idx)}")

n_unsafe = min(len(unsafe_idx), MAX_SAMPLES // 2)
n_safe   = min(len(safe_idx),   MAX_SAMPLES // 2)
rng = np.random.default_rng(42)
sel_safe   = rng.choice(safe_idx,   n_safe,   replace=False)
sel_unsafe = rng.choice(unsafe_idx, n_unsafe, replace=False)
sel_idx    = np.concatenate([sel_safe, sel_unsafe])
rng.shuffle(sel_idx)

obs_sel  = obs_arr[sel_idx]    # (N_sel, 12, 64, 64)
cost_sel = cost_arr[sel_idx]   # (N_sel,)
print(f"Selected {n_safe} safe + {n_unsafe} unsafe = {len(sel_idx)} total for probing")

# Encode in batches
print("Encoding observations...")
BATCH = 256
z_list = []
with torch.no_grad():
    for i in range(0, len(obs_sel), BATCH):
        batch = torch.tensor(obs_sel[i:i+BATCH]).to(DEVICE)
        z = model.encoder(batch)   # (B, z_dim)
        z_list.append(z.cpu().numpy())
Z = np.concatenate(z_list, axis=0)   # (N_sel, z_dim)
labels = cost_sel.astype(int)        # 0=safe, 1=unsafe

print(f"Latent shape: {Z.shape}")
print(f"Latent mean:  {Z.mean():.4f}")
print(f"Latent std:   {Z.std():.4f}")
print(f"Latent norm (avg per sample): {np.linalg.norm(Z, axis=1).mean():.4f}")

# ── 1. Collapse check ─────────────────────────────────────────────────────────
print("\n── Collapse check ──")
# If collapsed, all z vectors are nearly identical → tiny pairwise std
sample = Z[:200]
pairwise_dists = np.linalg.norm(sample[:, None] - sample[None, :], axis=-1)
print(f"Mean pairwise L2 distance: {pairwise_dists.mean():.4f}")
print(f"Std  pairwise L2 distance: {pairwise_dists.std():.4f}")
if pairwise_dists.mean() < 1.0:
    print("⚠  Possible collapse — mean pairwise distance is very small")
else:
    print("✓  No collapse detected")

# ── 2. PCA ────────────────────────────────────────────────────────────────────
print("\n── PCA ──")
scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)

pca = PCA(n_components=50)
Z_pca = pca.fit_transform(Z_scaled)
print(f"Variance explained by top-10 PCs: {pca.explained_variance_ratio_[:10].sum()*100:.1f}%")
print(f"Variance explained by top-50 PCs: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# PCA scatter: PC1 vs PC2, coloured by safe/unsafe
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (1, 2)]):
    for lab, color, marker, name in [(0, "#4C9BE8", "o", "safe"), (1, "#E84C4C", "^", "unsafe")]:
        mask = labels == lab
        ax.scatter(Z_pca[mask, pc_x], Z_pca[mask, pc_y],
                   c=color, marker=marker, s=12, alpha=0.5, label=name)
    ax.set_xlabel(f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)")
    ax.set_ylabel(f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)")
    ax.set_title(f"PCA: PC{pc_x+1} vs PC{pc_y+1}")
    ax.legend()

plt.suptitle("Latent Space PCA — Safe vs Unsafe", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "pca_safe_unsafe.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_DIR}/pca_safe_unsafe.png")

# Explained variance curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker="o", markersize=3)
ax.axhline(90, color="red", linestyle="--", label="90%")
ax.set_xlabel("Number of PCs"); ax.set_ylabel("Cumulative variance explained (%)")
ax.set_title("PCA Explained Variance"); ax.legend()
fig.savefig(OUT_DIR / "pca_variance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_DIR}/pca_variance.png")

# ── 3. t-SNE ─────────────────────────────────────────────────────────────────
print("\n── t-SNE (on top-50 PCs) ──")
n_tsne = min(2000, len(Z_pca))
idx_tsne = rng.choice(len(Z_pca), n_tsne, replace=False)
Z_tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000)\
             .fit_transform(Z_pca[idx_tsne])
lab_tsne = labels[idx_tsne]

fig, ax = plt.subplots(figsize=(8, 7))
for lab, color, marker, name in [(0, "#4C9BE8", "o", "safe"), (1, "#E84C4C", "^", "unsafe")]:
    mask = lab_tsne == lab
    ax.scatter(Z_tsne[mask, 0], Z_tsne[mask, 1],
               c=color, marker=marker, s=14, alpha=0.6, label=name)
ax.set_title("t-SNE of Latent Space — Safe vs Unsafe", fontsize=12, fontweight="bold")
ax.legend(); ax.axis("off")
fig.savefig(OUT_DIR / "tsne_safe_unsafe.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT_DIR}/tsne_safe_unsafe.png")

# ── 4. Linear separability ────────────────────────────────────────────────────
print("\n── Linear probe (logistic regression on z) ──")
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(Z_scaled, labels, test_size=0.2, random_state=42, stratify=labels)
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
print(classification_report(y_te, y_pred, target_names=["safe", "unsafe"]))
acc = (y_pred == y_te).mean()
print(f"Linear probe accuracy: {acc*100:.1f}%")
print("(>50% = encoder encodes some safety signal; ~98% = random baseline if data is 98% safe)")

# ── 5. Mean latent vectors: safe vs unsafe ────────────────────────────────────
print("\n── Mean latent per class ──")
z_safe   = Z[labels == 0]
z_unsafe = Z[labels == 1]
print(f"Mean safe   z norm: {np.linalg.norm(z_safe.mean(0)):.4f}")
print(f"Mean unsafe z norm: {np.linalg.norm(z_unsafe.mean(0)):.4f}")
mean_dist = np.linalg.norm(z_safe.mean(0) - z_unsafe.mean(0))
print(f"L2 distance between class means: {mean_dist:.4f}")
if mean_dist > 1.0:
    print("✓  Class means are separated in latent space")
else:
    print("⚠  Class means are close — bisimulation signal may be too weak")

print(f"\nAll outputs saved to {OUT_DIR}/")
