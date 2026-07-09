"""
Train and conformally calibrate the safety classifier.

Pipeline:
  1. Load trained SafeJEPA encoder (frozen).
  2. Encode all observations from the dataset into latent vectors z.
  3. Stratified split: 70% train, 15% calibration, 15% test.
  4. Train ObstacleMLP with signed hinge loss on train split.
  5. Conformal calibration on unsafe samples in the calibration split.
  6. Report test accuracy and save classifier + threshold.

Usage:
    python train_classifier.py
    CHECKPOINT=/mnt/t7shield/safe_lewm_rho1.pt python train_classifier.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, ".")
from safe_lewm.config import Config
from safe_lewm.model import SafeJEPA
from safe_lewm.dataset import load_trajectories
from safe_lewm.classifier import ObstacleMLP, hinge_loss
from safe_lewm.conformal import calibrate_classifier

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT      = os.environ.get("CHECKPOINT", "/mnt/t7shield/safe_lewm_rho1.pt")
DATA_PATH       = os.environ.get("DATA_PATH",  "/mnt/t7shield/safety_point_goal.pkl")
CLF_OUT         = os.environ.get("CLF_OUT",    "/mnt/t7shield/classifier.pt")
DELTA           = float(os.environ.get("DELTA", "0.1"))    # miscoverage level
HIDDEN_DIM      = 64
DEPTH           = 2
DROPOUT         = 0.1
MARGIN          = 1.0
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
EPOCHS          = 100
BATCH_SIZE      = 256
MAX_OBS         = 20000   # cap to avoid OOM
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
print(f"Encoder checkpoint: {CHECKPOINT}")

# ── 1. Load frozen encoder ────────────────────────────────────────────────────
print("\nLoading encoder...")
cfg = Config()
model = SafeJEPA(cfg).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False))
model.eval()
encoder = model.encoder
for p in encoder.parameters():
    p.requires_grad_(False)

# ── 2. Collect and encode observations ───────────────────────────────────────
print("\nLoading trajectories...")
trajs = load_trajectories(DATA_PATH)

print("Collecting observations...")
obs_list, cost_list = [], []
for traj in trajs:
    for t in range(len(traj["obs"])):
        obs_list.append(traj["obs"][t])
        cost_list.append(traj["cost"][t])
    if len(obs_list) >= MAX_OBS:
        break

obs_arr  = np.array(obs_list[:MAX_OBS], dtype=np.float32)
cost_arr = np.array(cost_list[:MAX_OBS], dtype=np.float32)
print(f"Total: {len(obs_arr)} obs | safe={int((cost_arr==0).sum())} | unsafe={int((cost_arr==1).sum())}")

print("Encoding with frozen encoder...")
BATCH = 256
z_list = []
with torch.no_grad():
    for i in range(0, len(obs_arr), BATCH):
        x = torch.tensor(obs_arr[i:i+BATCH]).to(DEVICE)
        z = encoder(x).cpu()
        z_list.append(z)
Z = torch.cat(z_list, dim=0)  # (N, z_dim)

# Normalize features by training-set mean/std (computed after split)
labels_bin = cost_arr.astype(int)  # 0=safe, 1=unsafe

# ── 3. Stratified split (70 / 15 / 15) ───────────────────────────────────────
print("\nSplitting data...")
rng = np.random.default_rng(42)
safe_idx   = np.where(labels_bin == 0)[0]
unsafe_idx = np.where(labels_bin == 1)[0]

def split_idx(idx, rng, train_frac=0.70, cal_frac=0.15):
    idx = rng.permutation(idx)
    n = len(idx)
    n_train = int(n * train_frac)
    n_cal   = int(n * cal_frac)
    return idx[:n_train], idx[n_train:n_train+n_cal], idx[n_train+n_cal:]

safe_tr, safe_cal, safe_te     = split_idx(safe_idx, rng)
unsafe_tr, unsafe_cal, unsafe_te = split_idx(unsafe_idx, rng)

tr_idx  = np.concatenate([safe_tr, unsafe_tr])
cal_idx = np.concatenate([safe_cal, unsafe_cal])
te_idx  = np.concatenate([safe_te, unsafe_te])

print(f"  Train:  {len(tr_idx)}  (safe={len(safe_tr)}, unsafe={len(unsafe_tr)})")
print(f"  Cal:    {len(cal_idx)} (safe={len(safe_cal)}, unsafe={len(unsafe_cal)})")
print(f"  Test:   {len(te_idx)}  (safe={len(safe_te)}, unsafe={len(unsafe_te)})")

# Normalize using training set statistics
Z_train = Z[tr_idx]
z_mean = Z_train.mean(0)
z_std  = Z_train.std(0).clamp(min=1e-6)

def normalize(z): return (z - z_mean) / z_std

Z_tr_n  = normalize(Z[tr_idx])
Z_cal_n = normalize(Z[cal_idx])
Z_te_n  = normalize(Z[te_idx])

# Hinge labels: +1 safe, -1 unsafe
def make_hinge_labels(idx):
    y = torch.tensor(labels_bin[idx], dtype=torch.float32)
    return torch.where(y == 0, torch.ones_like(y), -torch.ones_like(y))

y_tr  = make_hinge_labels(tr_idx)
y_cal = make_hinge_labels(cal_idx)
y_te  = make_hinge_labels(te_idx)

# ── 4. Train classifier ───────────────────────────────────────────────────────
print("\nTraining classifier...")
clf = ObstacleMLP(cfg.z_dim, HIDDEN_DIM, DEPTH, DROPOUT).to(DEVICE)
opt = torch.optim.AdamW(clf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

train_ds = TensorDataset(Z_tr_n, y_tr)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

best_val_loss = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    clf.train()
    total_loss = 0.0
    for z_b, y_b in train_loader:
        z_b, y_b = z_b.to(DEVICE), y_b.to(DEVICE)
        opt.zero_grad()
        scores = clf(z_b)
        loss = hinge_loss(scores, y_b, MARGIN)
        loss.backward()
        nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(z_b)
    avg_loss = total_loss / len(tr_idx)

    # Validation on calibration set
    clf.eval()
    with torch.no_grad():
        val_scores = clf(Z_cal_n.to(DEVICE)).cpu()
        val_loss = hinge_loss(val_scores, y_cal, MARGIN).item()
        # Accuracy: predict safe if score > 0
        preds = (val_scores > 0).float()
        cal_safe_labels = (y_cal > 0).float()
        acc = (preds == cal_safe_labels).float().mean().item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in clf.state_dict().items()}

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | val_acc={acc*100:.1f}%")

clf.load_state_dict(best_state)
print(f"Loaded best model (val_loss={best_val_loss:.4f})")

# ── 5. Test evaluation ────────────────────────────────────────────────────────
print("\nTest evaluation...")
clf.eval()
with torch.no_grad():
    te_scores = clf(Z_te_n.to(DEVICE)).cpu()
    te_preds  = (te_scores > 0).float()
    te_labels_bin = (y_te > 0).float()
    acc = (te_preds == te_labels_bin).float().mean().item()

    # Per-class accuracy
    safe_mask   = te_labels_bin == 1
    unsafe_mask = te_labels_bin == 0
    safe_acc   = (te_preds[safe_mask]   == 1).float().mean().item() if safe_mask.any() else 0.0
    unsafe_acc = (te_preds[unsafe_mask] == 0).float().mean().item() if unsafe_mask.any() else 0.0

print(f"  Overall accuracy:  {acc*100:.1f}%")
print(f"  Safe recall:       {safe_acc*100:.1f}%")
print(f"  Unsafe recall:     {unsafe_acc*100:.1f}%")

# ── 6. Conformal calibration (on unsafe calibration samples only) ─────────────
print("\nConformal calibration...")
unsafe_cal_mask = (y_cal < 0)   # these are obstacle/unsafe samples
Z_cal_unsafe = Z_cal_n[unsafe_cal_mask]
print(f"  Calibration unsafe samples: {len(Z_cal_unsafe)}")

safe_threshold = calibrate_classifier(clf, Z_cal_unsafe.to(DEVICE), delta=DELTA, device=DEVICE)

# ── 7. Save ───────────────────────────────────────────────────────────────────
print(f"\nSaving to {CLF_OUT}...")
torch.save({
    "classifier_state": clf.state_dict(),
    "z_mean": z_mean,
    "z_std": z_std,
    "safe_threshold": safe_threshold,
    "delta": DELTA,
    "z_dim": cfg.z_dim,
    "hidden_dim": HIDDEN_DIM,
    "depth": DEPTH,
    "dropout": DROPOUT,
}, CLF_OUT)

print(f"\nDone.")
print(f"  Classifier saved: {CLF_OUT}")
print(f"  Conformal threshold (delta={DELTA}): {safe_threshold:.4f}")
print(f"  Interpretation: at planning time, reject any latent z where NN(z) < {safe_threshold:.4f}")
