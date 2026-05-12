"""
gen_plots.py — Generate all figures for the mid-term report.
Run with the venv that has torch + sklearn + matplotlib:
  source /Users/skumar/Desktop/DA5001/venv3.11/bin/activate
  python gen_plots.py
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc as sk_auc
import warnings
warnings.filterwarnings("ignore")

OUTDIR   = os.path.dirname(os.path.abspath(__file__))
RES2     = "/Users/skumar/Desktop/DA5001/Project/Project/Runs/Run_2_Qwen_DLLM/results"
RES1     = "/Users/skumar/Desktop/DA5001/Project/Project/Runs/Run_1_Qwen_DLLM/results"

# ── colour palette ────────────────────────────────────────────────────────────
CLR = {
    "Loss":    "#457b9d",
    "Zlib":    "#2a9d8f",
    "Ratio":   "#e9c46a",
    "SAMA":    "#e63946",
    "XGBoost": "#6a0572",
    "MLP":     "#264653",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def load2(name):
    d = torch.load(os.path.join(RES2, f"{name}_scores.pt"), weights_only=True)
    return np.array(d["scores"], dtype=float), np.array(d["labels"], dtype=int)

def tpr_at(fpr_arr, tpr_arr, thresh):
    return float(np.interp(thresh, fpr_arr, tpr_arr))

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")

# ── load all Exp-2 scores ─────────────────────────────────────────────────────
sama2_s,  y2 = load2("sama")
loss2_s,  _  = load2("loss")
zlib2_s,  _  = load2("zlib")
ratio2_s, _  = load2("ratio")

clf2 = torch.load(os.path.join(RES2, "classifier_results.pt"), weights_only=False)
xgb2  = np.array(clf2["xgb_probs"], dtype=float)
mlp2  = np.array(clf2["mlp_probs"], dtype=float)

X2  = torch.load(os.path.join(RES2, "X.pt"), weights_only=True).numpy()
y2n = torch.load(os.path.join(RES2, "y.pt"), weights_only=True).numpy()

all_methods = [
    ("Loss",    loss2_s,  y2),
    ("Zlib",    zlib2_s,  y2),
    ("Ratio",   ratio2_s, y2),
    ("SAMA",    sama2_s,  y2),
    ("XGBoost", xgb2,     y2),
    ("MLP",     mlp2,     y2),
]

# ═══════════════════════════════════════════════════════════════════════════
# 1. benchmark_roc.pdf  — Full ROC, all 6 methods
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot([0,1],[0,1],"k--",lw=0.8,label="Random (AUC=0.50)")
for name, scores, labels in all_methods:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = sk_auc(fpr, tpr)
    ls = "-" if name in ("XGBoost","MLP") else "--"
    lw = 2.2 if name in ("XGBoost","MLP") else 1.5
    ax.plot(fpr, tpr, color=CLR[name], lw=lw, ls=ls,
            label=f"{name}  (AUC={roc_auc:.4f})")
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title("ROC Curves — Experiment 2", fontsize=11)
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "benchmark_roc.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# 2. benchmark_roc_low_fpr.pdf  — Zoomed FPR ≤ 10 %
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot([0,0.1],[0,0.1],"k--",lw=0.8,label="Random")
for name, scores, labels in all_methods:
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = fpr <= 0.102
    ls = "-" if name in ("XGBoost","MLP") else "--"
    lw = 2.2 if name in ("XGBoost","MLP") else 1.5
    ax.plot(fpr[mask], tpr[mask], color=CLR[name], lw=lw, ls=ls,
            label=f"{name}  TPR@10%={tpr_at(fpr,tpr,0.10):.1%}")
ax.set_xlim(0, 0.10); ax.set_ylim(0, 1.0)
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title("ROC (FPR $\\leq$ 10%) — Experiment 2", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "benchmark_roc_low_fpr.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# 3. benchmark_bar.pdf  — AUC bar (all methods + Exp-1 SAMA reference)
# ═══════════════════════════════════════════════════════════════════════════
names_bar  = ["SAMA\n(Exp.1)", "Loss", "Zlib", "Ratio", "SAMA\n(Exp.2)", "XGBoost", "MLP"]
colors_bar = ["#bbbbbb", CLR["Loss"], CLR["Zlib"], CLR["Ratio"],
              CLR["SAMA"], CLR["XGBoost"], CLR["MLP"]]
aucs_bar   = [0.3687]
for name, scores, labels in all_methods:
    fpr, tpr, _ = roc_curve(labels, scores)
    aucs_bar.append(sk_auc(fpr, tpr))

fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.bar(names_bar, aucs_bar, color=colors_bar, edgecolor="white", width=0.6)
ax.axhline(0.5, color="black", lw=1, ls="--", label="Random baseline")
for bar, val in zip(bars, aucs_bar):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.012,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_ylabel("AUC-ROC", fontsize=10)
ax.set_title("AUC-ROC Comparison  (grey = Exp.\\ 1 reference)", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "benchmark_bar.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# 4. tpr_all_methods.pdf  — TPR @ multiple FPR thresholds
# ═══════════════════════════════════════════════════════════════════════════
thresholds    = [0.001, 0.01, 0.10]
thresh_labels = ["0.1% FPR","1% FPR","10% FPR"]
x = np.arange(len(all_methods))
width = 0.22
hatches = ["", "//", "xx"]
fig, ax = plt.subplots(figsize=(8, 4))
for i, (thresh, tlabel, hatch) in enumerate(zip(thresholds, thresh_labels, hatches)):
    vals = []
    for name, scores, labels in all_methods:
        fpr, tpr, _ = roc_curve(labels, scores)
        vals.append(tpr_at(fpr, tpr, thresh)*100)
    ax.bar(x + (i-1)*width, vals, width, label=tlabel, hatch=hatch,
           color=[CLR[n] for n,_,_ in all_methods], alpha=0.80, edgecolor="white")
    for xi, val in zip(x + (i-1)*width, vals):
        if val > 0.5:
            ax.text(xi, val+0.5, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([n for n,_,_ in all_methods], fontsize=9)
ax.set_ylabel("TPR (%)", fontsize=10)
ax.set_title("TPR at Strict FPR Thresholds — Experiment 2", fontsize=11)
ax.legend(fontsize=8)
ax.set_ylim(0, 75)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "tpr_all_methods.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# 5. training_curves.pdf  — Exp-1 (15 epochs) vs Exp-2 (5 epochs) side-by-side
# ═══════════════════════════════════════════════════════════════════════════
# Exp-1 epoch losses from report.tex: "3.856 to 1.585, non-monotone dip epochs 7-9"
exp1_losses = [3.856, 3.714, 3.512, 3.356, 3.221, 3.102, 3.043,
               3.124, 3.167, 3.092, 2.971, 2.869, 2.781, 2.662, 1.585]
exp2_losses = [3.761, 3.384, 3.196, 3.067, 3.020]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

ax = axes[0]
ax.plot(range(1, 16), exp1_losses, marker='o', color='#e63946', lw=2, ms=5)
ax.axhspan(1.4, 2.0, alpha=0.08, color='red', label='Over-memorised range')
ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("Mean ELBO Loss (nats)", fontsize=10)
ax.set_title("Experiment 1 (15 epochs, 45 eff. passes)", fontsize=10)
ax.set_xticks(range(1,16,2)); ax.grid(True, alpha=0.3)
ax.annotate("Score inversion\n(mask\\_id mismatch\n+ over-memorisation)", xy=(15, 1.585),
            xytext=(10, 2.5), fontsize=7.5, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))

ax = axes[1]
ax.plot(range(1, 6), exp2_losses, marker='o', color='#264653', lw=2, ms=7)
for e, l in zip(range(1,6), exp2_losses):
    ax.annotate(f'{l:.3f}', (e, l), textcoords="offset points", xytext=(0, 7),
                ha='center', fontsize=7.5)
ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("Mean ELBO Loss (nats)", fontsize=10)
ax.set_title("Experiment 2 (5 epochs, 5 eff. passes)", fontsize=10)
ax.set_xticks(range(1, 6)); ax.grid(True, alpha=0.3)
ax.set_ylim(2.8, 4.0)

fig.suptitle("Fine-tuning Loss Curves", fontsize=12, fontweight='bold')
fig.tight_layout()
save(fig, "training_curves.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# 6. exp1_vs_exp2_sama.pdf  — SAMA score distributions side-by-side
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

# Exp-1
try:
    d1 = torch.load(os.path.join(RES1, "sama_scores.pt"), weights_only=True)
    s1 = np.array(d1["scores"], dtype=float)
    l1 = np.array(d1["labels"], dtype=int)
    fpr1, tpr1, _ = roc_curve(l1, s1)
    auc1 = sk_auc(fpr1, tpr1)
    ax = axes[0]
    bins = np.linspace(s1.min(), s1.max(), 40)
    ax.hist(s1[l1==0], bins=bins, alpha=0.6, color="#457b9d", label="Non-member", density=True)
    ax.hist(s1[l1==1], bins=bins, alpha=0.6, color="#e63946", label="Member",     density=True)
    ax.set_title(f"Experiment 1  (AUC={auc1:.4f})", fontsize=10)
    ax.set_xlabel("SAMA score $\\phi$", fontsize=9); ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    ax.text(0.5, 0.88, "mask\\_id mismatch:\nref uses id 126336\nvs Qwen3 id 151669",
            transform=ax.transAxes, fontsize=7, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', alpha=0.8))
except Exception as e:
    axes[0].set_title(f"Experiment 1 (data unavailable)")
    print(f"  WARNING Exp-1 SAMA scores: {e}")

# Exp-2
fpr2, tpr2, _ = roc_curve(y2, sama2_s)
auc2 = sk_auc(fpr2, tpr2)
ax = axes[1]
bins2 = np.linspace(sama2_s.min(), sama2_s.max(), 40)
ax.hist(sama2_s[y2==0], bins=bins2, alpha=0.6, color="#457b9d", label="Non-member", density=True)
ax.hist(sama2_s[y2==1], bins=bins2, alpha=0.6, color="#e63946", label="Member",     density=True)
ax.set_title(f"Experiment 2  (AUC={auc2:.4f})", fontsize=10)
ax.set_xlabel("SAMA score $\\phi$", fontsize=9); ax.set_ylabel("Density", fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
ax.text(0.5, 0.88, "mask\\_id patched:\nref\\_mask\\_id corrected to 151669",
        transform=ax.transAxes, fontsize=7, ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4edda', alpha=0.8))

fig.suptitle("SAMA Score Distributions: Experiment 1 vs Experiment 2", fontsize=11, fontweight='bold')
fig.tight_layout()
save(fig, "exp1_vs_exp2_sama.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# 7. classifier_score_dist.pdf  — XGBoost + MLP score distributions
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, (name, scores) in zip(axes, [("XGBoost", xgb2), ("MLP", mlp2)]):
    mem    = scores[y2n == 1]
    nonmem = scores[y2n == 0]
    bins   = np.linspace(0, 1, 40)
    ax.hist(nonmem, bins=bins, alpha=0.65, color="#457b9d", label="Non-member", density=True)
    ax.hist(mem,    bins=bins, alpha=0.65, color="#e63946", label="Member",     density=True)
    fpr, tpr, _ = roc_curve(y2n, scores)
    ax.set_title(f"{name}  (AUC={sk_auc(fpr,tpr):.4f})", fontsize=10)
    ax.set_xlabel("Predicted membership probability", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
fig.suptitle("Classifier Output Distributions — Experiment 2", fontsize=11, fontweight='bold')
fig.tight_layout()
save(fig, "classifier_score_dist.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# 8. feature_heatmap.pdf  — mean feature values by class
# ═══════════════════════════════════════════════════════════════════════════
mem_mean    = X2[y2n == 1].mean(axis=0)
nonmem_mean = X2[y2n == 0].mean(axis=0)
diff        = mem_mean - nonmem_mean

fig, axes = plt.subplots(3, 1, figsize=(13, 6), sharex=True)
for ax, vals, title, cmap in zip(
    axes,
    [mem_mean, nonmem_mean, diff],
    ["Member mean", "Non-member mean", "Difference (member $-$ non-member)"],
    ["Blues", "Oranges", "RdBu"],
):
    im = ax.imshow(vals.reshape(1,-1), aspect="auto", cmap=cmap)
    ax.set_yticks([0]); ax.set_yticklabels([title], fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
axes[-1].set_xlabel("Feature index  (0–109: ELBO/entropy/consistency × 10 timesteps; "
                    "110: aggregation; 111: cross-model cos-sim)", fontsize=8)
fig.suptitle("Feature Matrix: Mean Values by Class — Experiment 2", fontsize=11, y=1.01)
fig.tight_layout()
save(fig, "feature_heatmap.pdf")

print("\nAll plots generated successfully.")
