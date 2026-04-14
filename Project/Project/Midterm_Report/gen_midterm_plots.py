"""
gen_midterm_plots.py
====================
Generate all publication-quality figures for the Midterm Report.
Incorporates AttenMIA-style per-feature-group analysis (Hellinger distances,
KDE plots) plus ROC curves, TPR bars, score distributions, training curves,
and ablation plots.

Run with the DA5001 venv:
  /Users/skumar/Desktop/DA5001/venv3.11/bin/python3 gen_midterm_plots.py
"""

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve

# ── paths ─────────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
BASE2  = os.path.join(HERE, "../Logs/Run_2_Qwen_DLLM/results")
BASE1  = os.path.join(HERE, "../Logs/Run_1_Qwen_DLLM/results")
FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "lines.linewidth":   1.8,
})

COLORS = {
    "sama":   "#1f77b4",
    "loss":   "#ff7f0e",
    "zlib":   "#2ca02c",
    "ratio":  "#d62728",
    "xgb":    "#9467bd",
    "mlp":    "#8c564b",
    "member": "#2166ac",
    "nonmem": "#d73027",
}
METHOD_LABELS = {
    "sama":  "SAMA",
    "loss":  "Loss",
    "zlib":  "Zlib",
    "ratio": "Ratio",
    "xgb":   "DT-MIA (XGB)",
    "mlp":   "DT-MIA (MLP)",
}

def savefig(name):
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓  {name}")

def hellinger(p, q):
    """Hellinger distance between two sample arrays."""
    # estimate PDFs on shared grid
    lo = min(p.min(), q.min()); hi = max(p.max(), q.max())
    g  = np.linspace(lo, hi, 500)
    bw = "scott"
    try:
        kp = gaussian_kde(p, bw_method=bw)(g)
        kq = gaussian_kde(q, bw_method=bw)(g)
    except Exception:
        return float("nan")
    kp /= kp.sum(); kq /= kq.sum()
    return float(np.sqrt(0.5 * np.sum((np.sqrt(kp) - np.sqrt(kq))**2)))

# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data …")

# Run 2
r2 = {}
for name in ["sama", "loss", "zlib", "ratio"]:
    d = torch.load(f"{BASE2}/{name}_scores.pt", weights_only=True)
    r2[name] = {
        "scores": np.array(d["scores"], dtype=float),
        "labels": np.array(d["labels"], dtype=int),
    }
clf2  = torch.load(f"{BASE2}/classifier_results.pt", weights_only=False)
X2    = torch.load(f"{BASE2}/X.pt", weights_only=True).numpy()
y2    = torch.load(f"{BASE2}/y.pt", weights_only=True).numpy().astype(int)
xgb2p = np.array(clf2["xgb_probs"]); mlp2p = np.array(clf2["mlp_probs"])
y2c   = np.array(clf2["y_true"]).astype(int)

# Run 1
d1 = torch.load(f"{BASE1}/sama_scores.pt", weights_only=True)
r1_sama = {
    "scores": np.array(d1["scores"], dtype=float),
    "labels": np.array(d1["labels"], dtype=int),
}
clf1  = torch.load(f"{BASE1}/classifier_results.pt", weights_only=False)
X1    = torch.load(f"{BASE1}/X.pt", weights_only=True).numpy()
y1    = torch.load(f"{BASE1}/y.pt", weights_only=True).numpy().astype(int)
xgb1p = np.array(clf1["xgb_probs"]); mlp1p = np.array(clf1["mlp_probs"])
y1c   = np.array(clf1["y_true"]).astype(int)

# Feature layout (T=10, D=112)
T = 10
FEATURE_GROUPS = [
    ("ELBO traj.",       slice(0,   10)),
    ("dL/dt",            slice(11,  21)),
    ("d²L/dt²",          slice(21,  31)),
    ("Pred. entropy",    slice(31,  41)),
    ("Mask consist.",    slice(41,  51)),
    ("Hidden norms",     slice(51,  61)),
    ("Hidden cos-sim",   slice(61,  71)),
    ("Attn. entropy",    slice(71,  81)),
    ("Attn. crosslayer", slice(81,  91)),
    ("Attn. barycenter", slice(91,  101)),
    ("Attn. perturb.",   slice(101, 111)),
]

print("Data loaded. Generating figures …\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: Training loss curves (both runs)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 1: finetune_loss.pdf")
r1_losses = [3.8561,3.0240,2.8769,2.6962,2.4149,2.1070,2.1870,
             1.8002,2.0066,1.7610,1.7839,1.7662,1.6404,1.6536,1.5854]
r2_losses = [3.7610,3.3839,3.1962,3.0667,3.0196]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=False)
ax = axes[0]
ax.plot(range(1, 16), r1_losses, "o-", color="#1f77b4", ms=4, label="Run 1 (15 epochs)")
ax.axhline(r1_losses[-1], ls="--", color="#1f77b4", alpha=0.5, lw=1)
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean NLL Loss (nats)")
ax.set_title("Run 1 — 15 Epochs, 300 samples\nQwen3-0.6B MDLM")
ax.legend(); ax.set_xlim(1, 15)

ax = axes[1]
ax.plot(range(1, 6), r2_losses, "s-", color="#d62728", ms=4, label="Run 2 (5 epochs)")
ax.axhline(r2_losses[-1], ls="--", color="#d62728", alpha=0.5, lw=1)
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean NLL Loss (nats)")
ax.set_title("Run 2 — 5 Epochs, 1000 samples\nQwen3-0.6B MDLM")
ax.legend(); ax.set_xlim(1, 5)

fig.suptitle("Fine-tuning Loss Curves on OpenWebText Member Subset", fontsize=11, y=1.02)
plt.tight_layout()
savefig("finetune_loss.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: Memorization verification (ELBO gaps)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2: memorization_gap.pdf")
# Run 1: base_member=5.344, ft_member=0.707, member_gap=4.625, nonmem_gap=-0.130
# Run 2: base_member=4.594, ft_member=1.836, member_gap=2.766, nonmem_gap=1.797
groups = ["Run 1\n(300 samples)", "Run 2\n(1000 samples)"]
mem_gaps  = [4.625, 2.766]
nonm_gaps = [-0.130, 1.797]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# Left: ELBO loss comparison (base vs finetuned)
ax = axes[0]
x = np.arange(2)
w = 0.3
base_m  = [5.344, 4.594]
ft_m    = [0.707, 1.836]
base_nm = [5.214, 2.797]   # base_member - member_gap ≈ nonmember base
ft_nm   = [5.344, 4.593]   # ft_nonmember = base_nonmember - nonmem_gap (approx)

bars1 = ax.bar(x - w/2, base_m, w, label="Base model (member)", color="#4393c3", alpha=0.85)
bars2 = ax.bar(x + w/2, ft_m,   w, label="Finetuned (member)",  color="#d73027", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("Mean ELBO Loss (nats)")
ax.set_title("Base vs. Finetuned ELBO\non Member Samples")
ax.legend(fontsize=8)
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                          f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                          f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)

# Right: ELBO gaps (member vs nonmember)
ax = axes[1]
x2 = np.arange(2)
bg = ax.bar(x2 - w/2, mem_gaps,  w, label="Member gap",     color="#2ca02c", alpha=0.85)
rg = ax.bar(x2 + w/2, nonm_gaps, w, label="Non-member gap", color="#ff7f0e", alpha=0.85)
ax.axhline(0.05, ls="--", color="gray", lw=1, label="Required ≥ 0.05")
ax.set_xticks(x2); ax.set_xticklabels(groups)
ax.set_ylabel("ELBO Gap: Base − Finetuned (nats)")
ax.set_title("Memorization Verification\n(ELBO Gap)")
ax.legend(fontsize=8)
for bar in bg: ax.text(bar.get_x()+bar.get_width()/2, max(bar.get_height(),0)+0.05,
                       f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
for bar in rg: ax.text(bar.get_x()+bar.get_width()/2,
                       max(bar.get_height(),0)+0.05 if bar.get_height()>=0 else bar.get_height()-0.2,
                       f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)

fig.suptitle("Memorization Verification: Fine-tuned MDLM on OpenWebText", fontsize=11, y=1.02)
plt.tight_layout()
savefig("memorization_gap.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: ROC curves — all methods (Run 2)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3: roc_all_methods.pdf")
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

def plot_roc(ax, scores, labels, color, label, ls="-"):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax.plot(fpr, tpr, color=color, ls=ls, lw=1.8, label=f"{label} (AUC={auc:.3f})")

ax = axes[0]
for k in ["sama","loss","zlib","ratio"]:
    plot_roc(ax, r2[k]["scores"], r2[k]["labels"], COLORS[k], METHOD_LABELS[k])
plot_roc(ax, xgb2p, y2c, COLORS["xgb"], METHOD_LABELS["xgb"], ls="-")
plot_roc(ax, mlp2p, y2c, COLORS["mlp"], METHOD_LABELS["mlp"], ls="--")
ax.plot([0,1],[0,1],"k--",lw=0.8, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Methods (Run 2)")
ax.legend(fontsize=7.5, loc="lower right")

# Low FPR zoom
ax = axes[1]
xlim = 0.1
for k in ["sama","loss","zlib","ratio"]:
    fpr, tpr, _ = roc_curve(r2[k]["labels"], r2[k]["scores"])
    mask = fpr <= xlim
    ax.plot(fpr[mask], tpr[mask], color=COLORS[k], lw=1.8, label=METHOD_LABELS[k])
fpr, tpr, _ = roc_curve(y2c, xgb2p)
mask = fpr <= xlim
ax.plot(fpr[mask], tpr[mask], color=COLORS["xgb"], lw=1.8, label=METHOD_LABELS["xgb"])
fpr, tpr, _ = roc_curve(y2c, mlp2p)
mask = fpr <= xlim
ax.plot(fpr[mask], tpr[mask], color=COLORS["mlp"], lw=1.8, ls="--", label=METHOD_LABELS["mlp"])
ax.plot([0,xlim],[0,xlim],"k--",lw=0.8, alpha=0.5)
ax.set_xlim(0, xlim); ax.set_ylim(0, None)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Low-FPR Region (FPR ≤ 10%)")
ax.legend(fontsize=7.5, loc="upper left")

fig.suptitle("Membership Inference Attack Comparison — Qwen3-0.6B MDLM", fontsize=11, y=1.02)
plt.tight_layout()
savefig("roc_all_methods.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: AUC + TPR benchmark bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4: benchmark_bar.pdf")

methods = ["SAMA", "Loss", "Zlib", "Ratio", "DT-MIA\n(XGB)", "DT-MIA\n(MLP)"]
m_colors = [COLORS["sama"], COLORS["loss"], COLORS["zlib"], COLORS["ratio"],
            COLORS["xgb"], COLORS["mlp"]]

auc_vals = [
    roc_auc_score(r2["sama"]["labels"], r2["sama"]["scores"]),
    roc_auc_score(r2["loss"]["labels"], r2["loss"]["scores"]),
    roc_auc_score(r2["zlib"]["labels"], r2["zlib"]["scores"]),
    roc_auc_score(r2["ratio"]["labels"], r2["ratio"]["scores"]),
    roc_auc_score(y2c, xgb2p),
    roc_auc_score(y2c, mlp2p),
]

def get_tpr(scores, labels, fpr_thr):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(fpr_thr, fpr, tpr))

tpr10 = [
    get_tpr(r2["sama"]["scores"], r2["sama"]["labels"], 0.10),
    get_tpr(r2["loss"]["scores"], r2["loss"]["labels"], 0.10),
    get_tpr(r2["zlib"]["scores"], r2["zlib"]["labels"], 0.10),
    get_tpr(r2["ratio"]["scores"],r2["ratio"]["labels"],0.10),
    get_tpr(xgb2p, y2c, 0.10),
    get_tpr(mlp2p, y2c, 0.10),
]
tpr1 = [
    get_tpr(r2["sama"]["scores"], r2["sama"]["labels"], 0.01),
    get_tpr(r2["loss"]["scores"], r2["loss"]["labels"], 0.01),
    get_tpr(r2["zlib"]["scores"], r2["zlib"]["labels"], 0.01),
    get_tpr(r2["ratio"]["scores"],r2["ratio"]["labels"],0.01),
    get_tpr(xgb2p, y2c, 0.01),
    get_tpr(mlp2p, y2c, 0.01),
]
tpr01 = [
    get_tpr(r2["sama"]["scores"], r2["sama"]["labels"], 0.001),
    get_tpr(r2["loss"]["scores"], r2["loss"]["labels"], 0.001),
    get_tpr(r2["zlib"]["scores"], r2["zlib"]["labels"], 0.001),
    get_tpr(r2["ratio"]["scores"],r2["ratio"]["labels"],0.001),
    get_tpr(xgb2p, y2c, 0.001),
    get_tpr(mlp2p, y2c, 0.001),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(len(methods))
w = 0.55

ax = axes[0]
bars = ax.bar(x, auc_vals, w, color=m_colors, alpha=0.85, edgecolor="white", lw=0.5)
ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.7, label="Random (0.5)")
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("ROC AUC"); ax.set_ylim(0.45, 1.0)
ax.set_title("ROC AUC by Method (Run 2)")
for b, v in zip(bars, auc_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

ax = axes[1]
w2 = 0.22
bars1 = ax.bar(x - w2, [v*100 for v in tpr10], w2, label="TPR@10%FPR",
               color=[c for c in m_colors], alpha=0.9, edgecolor="white")
bars2 = ax.bar(x,       [v*100 for v in tpr1],  w2, label="TPR@1%FPR",
               color=[c for c in m_colors], alpha=0.65, edgecolor="white")
bars3 = ax.bar(x + w2, [v*100 for v in tpr01], w2, label="TPR@0.1%FPR",
               color=[c for c in m_colors], alpha=0.4, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("TPR (%)"); ax.set_title("TPR at Various FPR Thresholds (Run 2)")
ax.legend(fontsize=8)

fig.suptitle("Attack Performance Benchmark — Qwen3-0.6B MDLM (2000 samples)", fontsize=11, y=1.02)
plt.tight_layout()
savefig("benchmark_bar.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: Score distributions (member vs non-member)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5: score_distributions.pdf")
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

plot_configs = [
    ("sama",  r2["sama"]["scores"],  r2["sama"]["labels"]),
    ("loss",  r2["loss"]["scores"],  r2["loss"]["labels"]),
    ("zlib",  r2["zlib"]["scores"],  r2["zlib"]["labels"]),
    ("ratio", r2["ratio"]["scores"], r2["ratio"]["labels"]),
    ("xgb",   xgb2p, y2c),
    ("mlp",   mlp2p, y2c),
]

for ax, (key, scores, labels) in zip(axes, plot_configs):
    mem  = scores[labels == 1]
    nonm = scores[labels == 0]
    auc  = roc_auc_score(labels, scores)
    hd   = hellinger(mem, nonm)

    lo, hi = min(scores.min(), scores.min()), max(scores.max(), scores.max())
    grid = np.linspace(lo, hi, 300)

    try:
        kde_m = gaussian_kde(mem,  bw_method="scott")(grid)
        kde_n = gaussian_kde(nonm, bw_method="scott")(grid)
        ax.fill_between(grid, kde_m, alpha=0.35, color=COLORS["member"],  label=f"Member (n={len(mem)})")
        ax.fill_between(grid, kde_n, alpha=0.35, color=COLORS["nonmem"],  label=f"Non-member (n={len(nonm)})")
        ax.plot(grid, kde_m, color=COLORS["member"],  lw=1.5)
        ax.plot(grid, kde_n, color=COLORS["nonmem"],  lw=1.5)
    except Exception:
        ax.hist(mem,  bins=40, alpha=0.5, color=COLORS["member"],  density=True)
        ax.hist(nonm, bins=40, alpha=0.5, color=COLORS["nonmem"],  density=True)

    ax.set_title(f"{METHOD_LABELS[key]}\nAUC={auc:.3f}  HD={hd:.3f}")
    ax.set_xlabel("Score"); ax.set_ylabel("Density")
    ax.legend(fontsize=7)

fig.suptitle("Score Distributions: Member vs. Non-Member (Run 2, Qwen3-0.6B MDLM)", fontsize=11)
plt.tight_layout()
savefig("score_distributions.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6: Hellinger Distances per feature group (AttenMIA-style Table → Bar)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 6: hellinger_feature_groups.pdf")

mem_idx  = np.where(y2 == 1)[0]
nonm_idx = np.where(y2 == 0)[0]

# Group-level Hellinger: average HD across features in each group
group_hds = []
for gname, gslice in FEATURE_GROUPS:
    feats = X2[:, gslice]
    hds = []
    for col in range(feats.shape[1]):
        fm = feats[mem_idx, col]; fn = feats[nonm_idx, col]
        if fm.std() > 1e-8 and fn.std() > 1e-8:
            hds.append(hellinger(fm, fn))
    group_hds.append(np.nanmean(hds) if hds else 0.0)

# Baseline Hellinger (SAMA/Loss/Zlib/Ratio)
baseline_hds = {}
for k in ["sama","loss","zlib","ratio"]:
    mem  = r2[k]["scores"][r2[k]["labels"]==1]
    nonm = r2[k]["scores"][r2[k]["labels"]==0]
    baseline_hds[k] = hellinger(mem, nonm)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: feature group HD bar
ax = axes[0]
group_names = [g[0] for g in FEATURE_GROUPS]
c_attn  = "#9467bd"
c_traj  = "#1f77b4"
c_hid   = "#2ca02c"
bar_colors = [
    c_traj, c_traj, c_traj,   # ELBO, dL/dt, d²L/dt²
    "#ff7f0e", "#ff7f0e",       # pred entropy, mask consistency
    c_hid, c_hid,               # hidden norms, cosine
    c_attn, c_attn, c_attn, c_attn,  # attn entropy, crosslayer, barycenter, perturb
]
xg = np.arange(len(group_names))
bars = ax.bar(xg, group_hds, 0.65, color=bar_colors, edgecolor="white", lw=0.4, alpha=0.88)
ax.set_xticks(xg)
ax.set_xticklabels(group_names, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean Hellinger Distance")
ax.set_title("Feature Group Separability\n(Hellinger Distance, Run 2)")

# legend patches
patches = [
    mpatches.Patch(color=c_traj,  label="ELBO / Loss curve"),
    mpatches.Patch(color="#ff7f0e", label="Entropy / Consistency"),
    mpatches.Patch(color=c_hid,   label="Hidden states"),
    mpatches.Patch(color=c_attn,  label="Attention (AttenMIA-inspired)"),
]
ax.legend(handles=patches, fontsize=8, loc="upper right")

for bar, v in zip(bars, group_hds):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.003,
            f"{v:.3f}", ha='center', va='bottom', fontsize=7)

# Right: baseline vs DT-MIA composite HD
ax = axes[1]
bl_names = ["SAMA", "Loss", "Zlib", "Ratio"]
bl_hds   = [baseline_hds[k] for k in ["sama","loss","zlib","ratio"]]
bl_cols  = [COLORS[k] for k in ["sama","loss","zlib","ratio"]]

dt_hd_xgb = hellinger(xgb2p[y2c==1], xgb2p[y2c==0])
dt_hd_mlp = hellinger(mlp2p[y2c==1], mlp2p[y2c==0])
all_names = bl_names + ["DT-MIA\n(XGB)", "DT-MIA\n(MLP)"]
all_hds   = bl_hds + [dt_hd_xgb, dt_hd_mlp]
all_cols  = bl_cols + [COLORS["xgb"], COLORS["mlp"]]

xa = np.arange(len(all_names))
bars2 = ax.bar(xa, all_hds, 0.6, color=all_cols, edgecolor="white", lw=0.4, alpha=0.85)
ax.set_xticks(xa); ax.set_xticklabels(all_names, fontsize=9)
ax.set_ylabel("Hellinger Distance")
ax.set_title("Score-level Separability\nby Attack Method (Run 2)")
for bar, v in zip(bars2, all_hds):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.003,
            f"{v:.3f}", ha='center', va='bottom', fontsize=8)

fig.suptitle("Member vs. Non-Member Distributional Separability", fontsize=11, y=1.02)
plt.tight_layout()
savefig("hellinger_feature_groups.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7: KDE plots for key feature groups (AttenMIA Figure 3 style)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 7: feature_kde.pdf")

SHOW_GROUPS = [
    ("ELBO traj.",       slice(0, 10),   5),   # show t=5
    ("dL/dt",            slice(11, 21),  5),
    ("Pred. entropy",    slice(31, 41),  5),
    ("Mask consist.",    slice(41, 51),  5),
    ("Attn. entropy",    slice(71, 81),  5),
    ("Attn. crosslayer", slice(81, 91),  5),
    ("Attn. barycenter", slice(91, 101), 5),
    ("Attn. perturb.",   slice(101,111), 5),
]

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.flatten()

for ax, (gname, gslice, tidx) in zip(axes, SHOW_GROUPS):
    feats = X2[:, gslice]
    col   = min(tidx, feats.shape[1]-1)
    fm = feats[mem_idx, col]
    fn = feats[nonm_idx, col]
    hd = hellinger(fm, fn)

    lo, hi = np.percentile(np.concatenate([fm, fn]), [0.5, 99.5])
    grid = np.linspace(lo, hi, 300)
    try:
        kde_m = gaussian_kde(fm[np.isfinite(fm)], bw_method="scott")(grid)
        kde_n = gaussian_kde(fn[np.isfinite(fn)], bw_method="scott")(grid)
        ax.fill_between(grid, kde_m, alpha=0.40, color=COLORS["member"])
        ax.fill_between(grid, kde_n, alpha=0.40, color=COLORS["nonmem"])
        ax.plot(grid, kde_m, color=COLORS["member"],  lw=1.5, label="Member")
        ax.plot(grid, kde_n, color=COLORS["nonmem"],  lw=1.5, label="Non-member")
    except Exception:
        ax.hist(fm, bins=30, alpha=0.5, color=COLORS["member"], density=True)
        ax.hist(fn, bins=30, alpha=0.5, color=COLORS["nonmem"], density=True)

    ax.set_title(f"{gname}\n(t={col+1}, HD={hd:.3f})", fontsize=9)
    ax.set_xlabel("Feature value", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    if ax == axes[0]:
        ax.legend(fontsize=7)

fig.suptitle("KDE Distributions of Feature Groups: Member (blue) vs. Non-Member (orange)\nQwen3-0.6B MDLM, Run 2 (T=10 timesteps)", fontsize=11)
plt.tight_layout()
savefig("feature_kde.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8: Feature group mean separability across timesteps (layer-level analog)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 8: attn_feature_trajectory.pdf")

# For each attention group, compute HD at each timestep t=0..9
ATTN_GROUPS = [
    ("Attn. entropy",    slice(71, 81)),
    ("Attn. crosslayer", slice(81, 91)),
    ("Attn. barycenter", slice(91, 101)),
    ("Attn. perturb.",   slice(101,111)),
]
TRAJ_GROUPS = [
    ("ELBO traj.",    slice(0,  10)),
    ("dL/dt",         slice(11, 21)),
    ("Pred. entropy", slice(31, 41)),
    ("Mask consist.", slice(41, 51)),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ts = np.arange(1, T+1)

ax = axes[0]
attn_colors = ["#9467bd","#8c564b","#e377c2","#7f7f7f"]
for (gname, gslice), col in zip(ATTN_GROUPS, attn_colors):
    feats = X2[:, gslice]
    hds = []
    for t_idx in range(T):
        fm = feats[mem_idx, t_idx]
        fn = feats[nonm_idx, t_idx]
        if fm.std() > 1e-8 and fn.std() > 1e-8:
            hds.append(hellinger(fm, fn))
        else:
            hds.append(0.0)
    ax.plot(ts, hds, "o-", color=col, ms=4, label=gname)
ax.set_xlabel("Masking Timestep t"); ax.set_ylabel("Hellinger Distance")
ax.set_title("Attention Feature Separability\nAcross Masking Timesteps")
ax.legend(fontsize=8); ax.set_xlim(1, T)

ax = axes[1]
traj_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
for (gname, gslice), col in zip(TRAJ_GROUPS, traj_colors):
    feats = X2[:, gslice]
    hds = []
    for t_idx in range(T):
        fm = feats[mem_idx, t_idx]
        fn = feats[nonm_idx, t_idx]
        if fm.std() > 1e-8 and fn.std() > 1e-8:
            hds.append(hellinger(fm, fn))
        else:
            hds.append(0.0)
    ax.plot(ts, hds, "s-", color=col, ms=4, label=gname)
ax.set_xlabel("Masking Timestep t"); ax.set_ylabel("Hellinger Distance")
ax.set_title("ELBO/Entropy Feature Separability\nAcross Masking Timesteps")
ax.legend(fontsize=8); ax.set_xlim(1, T)

fig.suptitle("Per-Timestep Feature Separability: DLM Trajectory Analysis", fontsize=11, y=1.02)
plt.tight_layout()
savefig("attn_feature_trajectory.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 9: Run 1 vs Run 2 ablation
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 9: run_ablation.pdf")

# AUC values
r1_xgb_auc = roc_auc_score(y1c, xgb1p)
r1_mlp_auc = roc_auc_score(y1c, mlp1p)
r2_sama_auc = roc_auc_score(r2["sama"]["labels"], r2["sama"]["scores"])
r2_xgb_auc  = roc_auc_score(y2c, xgb2p)
r2_mlp_auc  = roc_auc_score(y2c, mlp2p)
r1_sama_auc = roc_auc_score(r1_sama["labels"], r1_sama["scores"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
metrics = ["SAMA\nAUC", "DT-MIA(XGB)\nAUC", "DT-MIA(MLP)\nAUC"]
r1_vals = [r1_sama_auc, r1_xgb_auc, r1_mlp_auc]
r2_vals = [r2_sama_auc, r2_xgb_auc, r2_mlp_auc]
xr = np.arange(len(metrics))
wr = 0.32
b1 = ax.bar(xr - wr/2, r1_vals, wr, label="Run 1 (300 samples, 128 tok)",
            color="#4393c3", alpha=0.85, edgecolor="white")
b2 = ax.bar(xr + wr/2, r2_vals, wr, label="Run 2 (1000 samples, 256 tok)",
            color="#d73027", alpha=0.85, edgecolor="white")
ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.6)
ax.set_xticks(xr); ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylabel("ROC AUC"); ax.set_ylim(0.3, 1.05)
ax.set_title("Run 1 vs. Run 2 — ROC AUC Comparison")
ax.legend(fontsize=8)
for b, v in zip(b1, r1_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
for b, v in zip(b2, r2_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

ax = axes[1]
# TPR @ 10% FPR for both runs
def get_tpr_arr(scores, labels, thrs):
    fpr, tpr, _ = roc_curve(labels, scores)
    return [float(np.interp(t, fpr, tpr)) for t in thrs]

thrs = [0.10, 0.01, 0.001]
thr_labels = ["TPR@10%", "TPR@1%", "TPR@0.1%"]

r1_xgb_tprs = get_tpr_arr(xgb1p, y1c, thrs)
r2_xgb_tprs = get_tpr_arr(xgb2p, y2c, thrs)
xt = np.arange(3)
wt = 0.32
bt1 = ax.bar(xt - wt/2, [v*100 for v in r1_xgb_tprs], wt,
             label="Run 1 — DT-MIA(XGB)", color="#4393c3", alpha=0.85, edgecolor="white")
bt2 = ax.bar(xt + wt/2, [v*100 for v in r2_xgb_tprs], wt,
             label="Run 2 — DT-MIA(XGB)", color="#d73027", alpha=0.85, edgecolor="white")
ax.set_xticks(xt); ax.set_xticklabels(thr_labels, fontsize=9)
ax.set_ylabel("TPR (%)"); ax.set_title("TPR at Low-FPR Thresholds\nDT-MIA (XGBoost)")
ax.legend(fontsize=8)
for b, v in zip(bt1, r1_xgb_tprs):
    ax.text(b.get_x()+b.get_width()/2, v*100+1, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=8)
for b, v in zip(bt2, r2_xgb_tprs):
    ax.text(b.get_x()+b.get_width()/2, v*100+1, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=8)

fig.suptitle("Ablation Study: Effect of Training Scale (Run 1 → Run 2)", fontsize=11, y=1.02)
plt.tight_layout()
savefig("run_ablation.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10: DT-MIA feature importance (XGBoost-based, grouped)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 10: feature_importance.pdf")

try:
    import xgboost as xgb_lib
    from sklearn.model_selection import StratifiedKFold

    clf_xgb = xgb_lib.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    clf_xgb.fit(X2, y2)
    importances = clf_xgb.feature_importances_  # shape [112]

    # Group importances
    all_groups = FEATURE_GROUPS + [("Cross-model\ncos-sim", slice(111, 112))]
    group_imps = []
    for gname, gslice in all_groups:
        group_imps.append((gname, importances[gslice].sum()))

    gnames = [g[0] for g in group_imps]
    gimps  = [g[1] for g in group_imps]
    sorted_idx = np.argsort(gimps)[::-1]

    gc_map = {}
    for g in group_imps:
        n = g[0]
        if "Attn" in n:       gc_map[n] = c_attn
        elif "Hidden" in n:   gc_map[n] = c_hid
        elif "ELBO" in n or "dL" in n or "d²" in n: gc_map[n] = c_traj
        else:                 gc_map[n] = "#ff7f0e"

    fig, ax = plt.subplots(figsize=(9, 4))
    xi = np.arange(len(gnames))
    bc = [gc_map.get(gnames[i], "#999") for i in sorted_idx]
    ax.bar(xi, [gimps[i] for i in sorted_idx], 0.65,
           color=bc, edgecolor="white", lw=0.4, alpha=0.88)
    ax.set_xticks(xi)
    ax.set_xticklabels([gnames[i] for i in sorted_idx], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative Feature Importance (XGBoost)")
    ax.set_title("DT-MIA Feature Group Importance\nXGBoost on Run 2 (2000 samples, 112 features)")

    patches_imp = [
        mpatches.Patch(color=c_traj,  label="ELBO / Loss curve"),
        mpatches.Patch(color="#ff7f0e", label="Entropy / Consistency"),
        mpatches.Patch(color=c_hid,   label="Hidden states"),
        mpatches.Patch(color=c_attn,  label="Attention (AttenMIA-inspired)"),
    ]
    ax.legend(handles=patches_imp, fontsize=8)
    plt.tight_layout()
    savefig("feature_importance.pdf")
    HAS_XGB = True
except Exception as e:
    print(f"  XGBoost not available or error: {e}")
    HAS_XGB = False

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11: DT-MIA classifier ROC curves + CI ribbons (Run 2)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 11: dtmia_roc_detail.pdf")

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# Full ROC with CI from classifier metrics
m_xgb = clf2["metrics_xgb"]
m_mlp = clf2["metrics_mlp"]

# Plot full curves
fpr_x, tpr_x, _ = roc_curve(y2c, xgb2p)
fpr_m, tpr_m, _ = roc_curve(y2c, mlp2p)

auc_xgb = m_xgb["auc_mean"]
auc_mlp = m_mlp["auc_mean"]

ax.plot(fpr_x, tpr_x, color=COLORS["xgb"], lw=2,
        label=f"DT-MIA XGB  AUC={auc_xgb:.4f} [{m_xgb['auc_lo']:.4f}–{m_xgb['auc_hi']:.4f}]")
ax.plot(fpr_m, tpr_m, color=COLORS["mlp"], lw=2, ls="--",
        label=f"DT-MIA MLP  AUC={auc_mlp:.4f} [{m_mlp['auc_lo']:.4f}–{m_mlp['auc_hi']:.4f}]")

# TPR CI markers at key FPRs
for fpr_thr, sym in [(0.10, "v"), (0.01, "^"), (0.001, "D")]:
    for metrics, proba, col in [(m_xgb, xgb2p, COLORS["xgb"]),
                                 (m_mlp, mlp2p, COLORS["mlp"])]:
        key = f"tpr_at_{fpr_thr}"
        if key in metrics:
            tpr_v   = metrics[key]
            tpr_lo  = metrics.get(f"{key}_lo", tpr_v)
            tpr_hi  = metrics.get(f"{key}_hi", tpr_v)
            ax.errorbar(fpr_thr, tpr_v,
                        yerr=[[tpr_v-tpr_lo],[tpr_hi-tpr_v]],
                        fmt=sym, color=col, ms=6, capsize=4, lw=1.5)

ax.plot([0,1],[0,1],"k--",lw=0.8, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("DT-MIA Classifier ROC with 95% CI\n(5-fold cross-validation, Run 2)")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
savefig("dtmia_roc_detail.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 12: Attack pipeline diagram (conceptual figure as matplotlib)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 12: pipeline_overview.pdf")

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
ax.set_title("DT-MIA Pipeline: Membership Inference on Masked Diffusion Language Models",
             fontsize=11, pad=10)

boxes = [
    (0.3, 1.0, 1.4, 1.4, "OpenWebText\n(Members +\nNon-members)", "#aec7e8"),
    (2.1, 1.0, 1.4, 1.4, "Fine-tune\nQwen3-0.6B\nMDLM", "#ffbb78"),
    (3.9, 1.0, 1.4, 1.4, "Mask trajectory\n(T=10 steps)\nForward passes", "#98df8a"),
    (5.7, 1.0, 1.4, 1.4, "Extract 112\nfeatures\n(ELBO+Attn+Hid)", "#ff9896"),
    (7.5, 1.0, 1.4, 1.4, "XGBoost/MLP\nClassifier\n(5-fold CV)", "#c5b0d5"),
    (9.1, 1.0, 0.8, 1.4, "Member\n/ Non-\nmember", "#c49c94"),
]

for x, y, w, h, txt, col in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.1", linewidth=1.2,
                                    edgecolor="#333", facecolor=col, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=8, fontweight="bold")

# arrows
arrow_xs = [(1.7, 2.1), (3.5, 3.9), (5.3, 5.7), (7.1, 7.5), (9.0, 9.1)]
for (x1, x2) in arrow_xs:
    ax.annotate("", xy=(x2, 1.7), xytext=(x1, 1.7),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

# SAMA parallel box
rect2 = mpatches.FancyBboxPatch((3.9, 0.0), 1.4, 0.8,
                                  boxstyle="round,pad=0.1", linewidth=1,
                                  edgecolor="#1f77b4", facecolor="#aec7e8", alpha=0.8,
                                  linestyle="--")
ax.add_patch(rect2)
ax.text(4.6, 0.4, "SAMA\n(black-box ref.)", ha="center", va="center", fontsize=7.5)
ax.annotate("", xy=(4.6, 0.8), xytext=(4.6, 1.0),
            arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.2, linestyle="dashed"))

plt.tight_layout()
savefig("pipeline_overview.pdf")

print("\n✅ All figures saved to:", FIGDIR)
