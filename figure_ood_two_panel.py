"""
Two-panel figure: OOD score mean ± std for ρ=1 and ρ=0 separately.

Reads eval_wm_viz/ood_records.npz produced by evaluate_ood_generalization.py.
Usage:
    python figure_ood_two_panel.py
    OUT_DIR=eval_wm_viz python figure_ood_two_panel.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.environ.get("OUT_DIR", "eval_wm_viz")
data    = np.load(f"{OUT_DIR}/ood_records.npz")
rec_in  = data["rec_in"]   # (N, 3): dist, score_reg, score_noreg
rec_ood = data["rec_ood"]

# ── Bins (same as evaluate_ood_generalization.py) ─────────────────────────────
MAX_DIST = 3.0
bins = np.concatenate([
    np.linspace(0, 1.0, 15),
    np.linspace(1.0, MAX_DIST, 10)[1:],
])
bc = 0.5 * (bins[:-1] + bins[1:])

def bin_scores(records, col):
    means, stds = [], []
    for i in range(len(bins) - 1):
        mask = (records[:, 0] >= bins[i]) & (records[:, 0] < bins[i + 1])
        s = records[mask, col]
        means.append(s.mean() if len(s) >= 3 else np.nan)
        stds.append(s.std()   if len(s) >= 3 else np.nan)
    return np.array(means), np.array(stds)

mr_in,  sr_in  = bin_scores(rec_in,  1)   # ρ=1 in-dist
mn_in,  sn_in  = bin_scores(rec_in,  2)   # ρ=0 in-dist
mr_ood, sr_ood = bin_scores(rec_ood, 1)   # ρ=1 OOD
mn_ood, sn_ood = bin_scores(rec_ood, 2)   # ρ=0 OOD

# ── Colors ────────────────────────────────────────────────────────────────────
C_IN  = "#2166ac"   # blue  — in-dist
C_OOD = "#74add1"   # light blue — OOD

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("OOD Score: Mean ± Std vs Distance to Nearest Hazard\n"
             "In-distribution (solid) vs Novel layouts (dashed)",
             fontsize=12)

panel_cfg = [
    # (col_in, col_ood, mean_in, std_in, mean_ood, std_ood, title, color_in, color_ood)
    (mr_in, sr_in, mr_ood, sr_ood, "ρ=1  (with regularization)", "#2166ac", "#74add1"),
    (mn_in, sn_in, mn_ood, sn_ood, "ρ=0  (no regularization)",   "#d73027", "#f46d43"),
]

for ax, (m_in, s_in, m_ood, s_ood, title, c_in, c_ood) in zip(axes, panel_cfg):
    v_in  = ~np.isnan(m_in)
    v_ood = ~np.isnan(m_ood)

    # In-distribution: mean line + std band
    ax.plot(bc[v_in], m_in[v_in], "-o", color=c_in, markersize=4,
            linewidth=2, label="In-dist mean")
    ax.fill_between(bc[v_in],
                    m_in[v_in] - s_in[v_in],
                    m_in[v_in] + s_in[v_in],
                    alpha=0.25, color=c_in, label="In-dist ±1 std")

    # OOD: mean line + std band
    ax.plot(bc[v_ood], m_ood[v_ood], "--o", color=c_ood, markersize=4,
            linewidth=2, label="OOD mean")
    ax.fill_between(bc[v_ood],
                    m_ood[v_ood] - s_ood[v_ood],
                    m_ood[v_ood] + s_ood[v_ood],
                    alpha=0.18, color=c_ood, label="OOD ±1 std")

    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Distance to nearest hazard (m)", fontsize=10)
    ax.set_ylabel("Safety score", fontsize=10)
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f"{OUT_DIR}/ood_two_panel.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
