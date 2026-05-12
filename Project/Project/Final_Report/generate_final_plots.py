"""
generate_final_plots.py
Generates all paper-ready figures for:
"Membership Inference Attacks on Discrete Diffusion Language Models"
Outputs go to ./Plots/
"""
import os, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
PROJ      = os.path.join(HERE, "..")
RUN3_BASE = os.path.join(PROJ, "Runs", "Run_3_Full_MIMIR",  "results")
RUN4_BASE = os.path.join(PROJ, "Runs", "Run_4_Shadow_MIA",  "shadow_results")
RUN1_BASE = os.path.join(PROJ, "Runs", "Run_1_Qwen_DLLM",   "results")
OUT       = os.path.join(HERE, "Plots")
os.makedirs(OUT, exist_ok=True)

DOMAINS = ["arxiv", "github", "hackernews", "pile_cc", "pubmed_central", "wikipedia"]
DOMAIN_LABELS = {
    "arxiv":          "arXiv",
    "github":         "GitHub",
    "hackernews":     "HackerNews",
    "pile_cc":        "Pile-CC",
    "pubmed_central": "PubMed",
    "wikipedia":      "Wikipedia",
}

# ---------------------------------------------------------------------------
# NeurIPS-compatible style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "figure.dpi":       300,
    "font.family":      "serif",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

PALETTE = {
    "Loss":     "#457b9d",
    "Zlib":     "#2a9d8f",
    "Ratio":    "#e9c46a",
    "SAMA":     "#e63946",
    "XGBoost":  "#6a0572",
    "LightGBM": "#1d3557",
    "MLP":      "#264653",
    "Shadow":   "#f4a261",
    "Oracle":   "#2c7bb6",
    "ELBO+H":   "#43aa8b",
    "Attn":     "#c9ada7",
    "Pruned":   "#f8961e",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_pt(path, silent=False):
    if not os.path.exists(path):
        if not silent:
            print(f"  [WARN] missing: {path}")
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def compute_roc(scores, labels):
    sc = np.array(scores.float() if hasattr(scores, "float") else scores)
    lb = np.array(labels.long()  if hasattr(labels,  "long")  else labels)
    fpr, tpr, _ = roc_curve(lb, sc)
    return fpr, tpr, auc(fpr, tpr)


def compute_pr(scores, labels):
    sc = np.array(scores.float() if hasattr(scores, "float") else scores)
    lb = np.array(labels.long()  if hasattr(labels,  "long")  else labels)
    prec, rec, _ = precision_recall_curve(lb, sc)
    ap = average_precision_score(lb, sc)
    return rec, prec, ap


def tpr_at_fpr(fpr_arr, tpr_arr, threshold):
    return float(np.interp(threshold, fpr_arr, tpr_arr))


def load_run3_domain(domain):
    base = os.path.join(RUN3_BASE, domain)
    cls  = load_pt(os.path.join(base, "classifier_results.pt"))
    sama = load_pt(os.path.join(base, "sama_scores.pt"))
    loss = load_pt(os.path.join(base, "loss_scores.pt"))
    zlib = load_pt(os.path.join(base, "zlib_scores.pt"))
    ratio= load_pt(os.path.join(base, "ratio_scores.pt"))
    return cls, sama, loss, zlib, ratio


def get_scores_labels(d, prefix=""):
    if d is None:
        return None, None
    if "scores" in d:
        return d["scores"], d["labels"]
    if f"{prefix}probs" in d:
        return d[f"{prefix}probs"], d["y_true"]
    return None, None


# ---------------------------------------------------------------------------
# Fig 1 — Conceptual overview (matplotlib schematic, no data)
# ---------------------------------------------------------------------------
def plot_fig1_concept():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    fig.subplots_adjust(wspace=0.35)

    # Panel A: MDLM masking process
    ax = axes[0]
    ax.set_xlim(0, 4); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("(a) MDLM Masking", fontsize=8, fontweight="bold")

    tokens_raw   = ["The", "cat", "sat", "on"]
    tokens_t1    = ["The", "[M]", "sat", "on"]
    tokens_t2    = ["[M]", "[M]", "sat", "[M]"]
    tokens_t3    = ["[M]", "[M]", "[M]", "[M]"]

    def draw_tokens(ax, tokens, y, color):
        for i, tok in enumerate(tokens):
            c = "#d62828" if tok == "[M]" else "#e8f4f8"
            rect = mpatches.FancyBboxPatch(
                (i * 0.95 + 0.02, y - 0.22), 0.88, 0.44,
                boxstyle="round,pad=0.02", facecolor=c,
                edgecolor="#333333", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(i * 0.95 + 0.46, y, tok, ha="center", va="center",
                    fontsize=5.5, color="white" if tok == "[M]" else "#222222")

    for row, (toks, y, lbl) in enumerate([
        (tokens_raw, 4.2, r"$x_0$"),
        (tokens_t1,  3.1, r"$x_{t_1}$"),
        (tokens_t2,  2.0, r"$x_{t_2}$"),
        (tokens_t3,  0.9, r"$x_{t_3}$"),
    ]):
        draw_tokens(ax, toks, y, None)
        ax.text(-0.25, y, lbl, ha="right", va="center", fontsize=7, style="italic")
        if row < 3:
            ax.annotate("", xy=(-0.05, y - 0.55), xytext=(-0.05, y - 0.3),
                        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

    # Panel B: ELBO trajectory curves
    ax = axes[1]
    ax.set_title("(b) ELBO Trajectory Signal", fontsize=8, fontweight="bold")
    t = np.array([0.05, 0.20, 0.35, 0.50])
    member_elbo     = np.array([1.2, 2.1, 3.0, 3.6])
    nonmember_elbo  = np.array([2.0, 2.9, 3.9, 4.5])
    ax.plot(t, member_elbo,    "o-", color=PALETTE["XGBoost"], lw=1.8,
            label="Member",     markersize=4)
    ax.plot(t, nonmember_elbo, "s--", color=PALETTE["SAMA"], lw=1.8,
            label="Non-member", markersize=4)
    ax.fill_between(t, member_elbo, nonmember_elbo, alpha=0.15, color="gray")
    ax.set_xlabel(r"Masking ratio $\alpha$", fontsize=8)
    ax.set_ylabel("ELBO (nats)", fontsize=8)
    ax.set_xticks(t)
    ax.legend(fontsize=7, loc="upper left")

    # Panel C: Feature → Classifier → Decision
    ax = axes[2]
    ax.set_xlim(0, 4); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("(c) Attack Pipeline", fontsize=8, fontweight="bold")

    boxes = [
        (2.0, 4.3, "46-dim\nFeatures",  "#adb5bd"),
        (2.0, 3.0, "XGBoost\nClassifier","#6a0572"),
        (1.0, 1.5, "Member",            "#2c7bb6"),
        (3.0, 1.5, "Non-member",        "#e63946"),
    ]
    for x, y, label, color in boxes:
        tc = "white" if color not in ("#adb5bd",) else "#222222"
        rect = mpatches.FancyBboxPatch(
            (x - 0.85, y - 0.35), 1.7, 0.7,
            boxstyle="round,pad=0.04", facecolor=color,
            edgecolor="#333333", linewidth=0.7)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=6.5, color=tc, fontweight="bold")

    ax.annotate("", xy=(2.0, 3.38), xytext=(2.0, 3.97),
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.9))
    ax.annotate("", xy=(1.0, 1.88), xytext=(1.7, 2.63),
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.9))
    ax.annotate("", xy=(3.0, 1.88), xytext=(2.3, 2.63),
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.9))
    ax.text(2.0, 0.5, "p > 0.5?", ha="center", fontsize=7, style="italic", color="#555")

    fig.savefig(os.path.join(OUT, "fig1_concept.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig1_concept.pdf")


# ---------------------------------------------------------------------------
# Fig 2 — AUC heatmap (6 domains × 6 methods)
# ---------------------------------------------------------------------------
def plot_fig2_auc_heatmap():
    methods = ["Loss", "Zlib", "Ratio", "SAMA", "XGBoost", "MLP"]
    data = np.zeros((len(methods), len(DOMAINS)))

    for j, dom in enumerate(DOMAINS):
        cls, sama, loss, zlib, ratio = load_run3_domain(dom)
        if cls is None: continue
        y = cls["y_true"]
        for i, (name, sc) in enumerate([
            ("Loss",    loss["scores"] if loss else None),
            ("Zlib",    zlib["scores"] if zlib else None),
            ("Ratio",   ratio["scores"] if ratio else None),
            ("SAMA",    sama["scores"] if sama else None),
            ("XGBoost", cls["xgb_probs"]),
            ("MLP",     cls["mlp_probs"]),
        ]):
            if sc is not None:
                _, _, a = compute_roc(sc, y)
                data[i, j] = a

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    dom_labels = [DOMAIN_LABELS[d] for d in DOMAINS]
    sns.heatmap(
        data, annot=True, fmt=".3f",
        xticklabels=dom_labels, yticklabels=methods,
        cmap="RdYlGn", vmin=0.5, vmax=1.0,
        linewidths=0.4, linecolor="#cccccc",
        ax=ax, cbar_kws={"label": "AUC-ROC", "shrink": 0.85},
        annot_kws={"size": 7.5},
    )
    ax.set_title("AUC-ROC across Methods and Domains", fontsize=9, fontweight="bold")
    ax.set_xlabel("Domain", fontsize=8)
    ax.set_ylabel("Attack Method", fontsize=8)
    # 45 deg rotation with right-alignment avoids overlap on long names like HackerNews
    ax.set_xticklabels(dom_labels, rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_auc_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig2_auc_heatmap.pdf")


# ---------------------------------------------------------------------------
# Fig 3 — Aggregate ROC curves (mean ± std across domains)
# ---------------------------------------------------------------------------
def plot_fig3_roc_aggregate():
    methods_cfg = [
        ("Loss",    "#457b9d", "-"),
        ("Zlib",    "#2a9d8f", "--"),
        ("Ratio",   "#e9c46a", ":"),
        ("SAMA",    "#e63946", "-."),
        ("XGBoost", "#6a0572", "-"),
        ("MLP",     "#264653", "--"),
    ]

    common_fpr = np.linspace(0, 1, 500)
    tpr_collections = {m[0]: [] for m in methods_cfg}

    for dom in DOMAINS:
        cls, sama, loss, zlib, ratio = load_run3_domain(dom)
        if cls is None: continue
        y = cls["y_true"]
        for name, sc in [
            ("Loss",    loss["scores"] if loss else None),
            ("Zlib",    zlib["scores"] if zlib else None),
            ("Ratio",   ratio["scores"] if ratio else None),
            ("SAMA",    sama["scores"] if sama else None),
            ("XGBoost", cls["xgb_probs"]),
            ("MLP",     cls["mlp_probs"]),
        ]:
            if sc is not None:
                fpr, tpr, _ = compute_roc(sc, y)
                interp = np.interp(common_fpr, fpr, tpr)
                tpr_collections[name].append(interp)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5, label="Random (AUC=0.5)")

    for name, color, ls in methods_cfg:
        tprs = np.array(tpr_collections[name])
        if len(tprs) == 0: continue
        mean_tpr = tprs.mean(0)
        std_tpr  = tprs.std(0)
        mean_auc = auc(common_fpr, mean_tpr)
        ax.plot(common_fpr, mean_tpr, color=color, ls=ls, lw=1.6,
                label=f"{name} ({mean_auc:.3f})")
        ax.fill_between(common_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        alpha=0.10, color=color)

    ax.set_xlabel("False Positive Rate", fontsize=8)
    ax.set_ylabel("True Positive Rate", fontsize=8)
    ax.set_title("Aggregate ROC Curves (mean ± std, 6 domains)", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_roc_aggregate.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig3_roc_aggregate.pdf")


# ---------------------------------------------------------------------------
# Fig 4 — TPR @ 0.1% FPR and 1% FPR two-panel bar chart
# ---------------------------------------------------------------------------
def plot_fig4_tpr_bar():
    methods    = ["Loss", "Zlib", "SAMA", "XGBoost", "MLP"]
    colors     = [PALETTE["Loss"], PALETTE["Zlib"], PALETTE["SAMA"],
                  PALETTE["XGBoost"], PALETTE["MLP"]]
    dom_labels = [DOMAIN_LABELS[d] for d in DOMAINS]
    thresholds = [0.001, 0.01]
    panel_titles = ["TPR @ 0.1% FPR", "TPR @ 1% FPR"]

    # collect per-threshold data
    all_data = []
    for thr in thresholds:
        data = {m: [] for m in methods}
        for dom in DOMAINS:
            cls, sama, loss, zlib, ratio = load_run3_domain(dom)
            if cls is None:
                for m in methods: data[m].append(0)
                continue
            y = cls["y_true"]
            for name, sc in [
                ("Loss",    loss["scores"] if loss else None),
                ("Zlib",    zlib["scores"] if zlib else None),
                ("SAMA",    sama["scores"] if sama else None),
                ("XGBoost", cls["xgb_probs"]),
                ("MLP",     cls["mlp_probs"]),
            ]:
                if sc is not None:
                    fpr, tpr, _ = compute_roc(sc, y)
                    data[name].append(tpr_at_fpr(fpr, tpr, thr))
                else:
                    data[name].append(0)
        all_data.append(data)

    x     = np.arange(len(DOMAINS))
    width = 0.14
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)

    for ax, data, title in zip(axes, all_data, panel_titles):
        for k, (name, color) in enumerate(zip(methods, colors)):
            offset = (k - len(methods) / 2 + 0.5) * width
            ax.bar(x + offset, [v * 100 for v in data[name]],
                   width, label=name, color=color, alpha=0.87,
                   edgecolor="white", lw=0.4)
        ax.set_xlabel("Domain", fontsize=8)
        ax.set_ylabel("TPR (%)", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(dom_labels, rotation=22, ha="right", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_ylim(0, None)

    # shared legend above both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 1.04),
               framealpha=0.9, handlelength=1.2,
               handletextpad=0.4, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_tpr_bar.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig4_tpr_bar.pdf")


# ---------------------------------------------------------------------------
# Fig 5 — Feature ablation (LOSO AUC drop, averaged across domains)
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "elbo_traj":        "ELBO Trajectory",
    "pred_entropy":     "Pred. Entropy",
    "hidden_norms":     "Hidden Norms",
    "mask_consistency": "Mask Consistency",
    "cross_model_cos":  "Cross-model Cosine",
    "hidden_cosine":    "Hidden Cosine",
    "dldt":             r"$dL/dt$",
    "d2ldt2":           r"$d^2L/dt^2$",
    "elbo_var":         "ELBO Variance",
    "attn_entropy":     "Attn Entropy",
    "attn_crosslayer":  "Attn Cross-layer",
    "attn_barycenter":  "Attn Barycenter",
    "attn_perturbation":"Attn Perturbation",
}

def plot_fig5_ablation():
    drops_per_group = {}
    solos_per_group = {}
    for dom in DOMAINS:
        abl = load_pt(os.path.join(RUN3_BASE, dom, "ablation_results.pt"), silent=True)
        if abl is None: continue
        for grp, vals in abl.items():
            drops_per_group.setdefault(grp, []).append(float(vals["auc_drop"]))
            solos_per_group.setdefault(grp, []).append(float(vals["solo_auc"]))

    mean_drops = {g: np.mean(v) for g, v in drops_per_group.items()}
    sorted_groups = sorted(mean_drops, key=lambda g: mean_drops[g], reverse=True)

    labels = [FEATURE_LABELS.get(g, g) for g in sorted_groups]
    values = [mean_drops[g] for g in sorted_groups]
    solos  = [np.mean(solos_per_group.get(g, [0])) for g in sorted_groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    cmap = plt.cm.RdYlGn_r
    norm_vals = (np.array(values) - min(values)) / (max(values) - min(values) + 1e-9)

    # LOSO AUC drop
    bars1 = ax1.barh(range(len(labels)), values,
                     color=[cmap(v) for v in norm_vals], edgecolor="#555", lw=0.4)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean LOSO AUC Drop", fontsize=8)
    ax1.set_title("Feature Importance\n(LOSO AUC Drop)", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars1, values):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=6.5)

    # Solo AUC
    bars2 = ax2.barh(range(len(labels)), solos,
                     color=[cmap(v) for v in norm_vals], edgecolor="#555", lw=0.4)
    ax2.set_xlabel("Mean Solo AUC", fontsize=8)
    ax2.set_title("Solo Feature AUC\n(Single Group)", fontsize=8.5, fontweight="bold")
    ax2.axvline(0.5, color="red", lw=0.8, ls="--", alpha=0.7, label="Random (0.5)")
    ax2.legend(fontsize=7)
    for bar, val in zip(bars2, solos):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=6.5)

    fig.savefig(os.path.join(OUT, "fig5_ablation.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig5_ablation.pdf")


# ---------------------------------------------------------------------------
# Fig 6 — Shadow model comparison
# ---------------------------------------------------------------------------
def plot_fig6_shadow():
    cond_keys   = ["A_oracle", "B_full46", "C_elbo_entropy8", "D_attn16", "E_pruned30"]
    cond_labels = ["Oracle (A)", "Shadow-46 (B)", "ELBO+H (C)", "Attn-only (D)", "Pruned-30 (E)"]
    cond_colors = [PALETTE["Oracle"], PALETTE["Shadow"], PALETTE["ELBO+H"],
                   PALETTE["Attn"], PALETTE["Pruned"]]

    auc_table = {k: [] for k in cond_keys}
    sama_aucs  = []

    for dom in DOMAINS:
        tr  = load_pt(os.path.join(RUN4_BASE, dom, "transfer_results.pt"), silent=True)
        cls = load_pt(os.path.join(RUN3_BASE, dom, "classifier_results.pt"), silent=True)
        sama= load_pt(os.path.join(RUN3_BASE, dom, "sama_scores.pt"), silent=True)

        if tr is None or cls is None: continue
        y = cls["y_true"]
        if sama:
            _, _, sa = compute_roc(sama["scores"], y)
            sama_aucs.append(sa)

        for ck in cond_keys:
            cond = tr["conditions"].get(ck)
            if cond and "probs" in cond:
                _, _, a = compute_roc(torch.tensor(cond["probs"]), y)
                auc_table[ck].append(a)
            else:
                auc_table[ck].append(np.nan)

    dom_labels = [DOMAIN_LABELS[d] for d in DOMAINS]
    x = np.arange(len(DOMAINS))
    width = 0.12
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    for k, (ck, label, color) in enumerate(zip(cond_keys, cond_labels, cond_colors)):
        vals = auc_table[ck]
        offset = (k - len(cond_keys) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color,
               alpha=0.88, edgecolor="white", lw=0.5)

    # SAMA per-domain reference lines
    import matplotlib.lines as mlines
    if sama_aucs:
        for j, (xp, sa) in enumerate(zip(x, sama_aucs)):
            lw_span = (len(cond_keys) / 2) * width + 0.04
            ax.hlines(sa, xp - lw_span, xp + lw_span,
                      colors=PALETTE["SAMA"], lw=1.5, ls="--", zorder=5)

    sama_line = mlines.Line2D([], [], color=PALETTE["SAMA"], lw=1.5, ls="--",
                              label="SAMA (baseline)")
    handles, labels_l = ax.get_legend_handles_labels()

    # Single-column legend anchored outside plot to avoid bar overlap
    ax.legend(handles + [sama_line], labels_l + ["SAMA (baseline)"],
              fontsize=7.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), framealpha=0.95,
              handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

    ax.set_xlabel("Domain", fontsize=8, labelpad=2)
    ax.set_ylabel("AUC-ROC", fontsize=8)
    ax.set_title("Shadow Model Transfer: AUC by Condition and Domain", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dom_labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylim(0.45, 1.05)
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(os.path.join(OUT, "fig6_shadow.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig6_shadow.pdf")


# ---------------------------------------------------------------------------
# Fig 7 — ELBO gap per domain
# ---------------------------------------------------------------------------
def plot_fig7_elbo_gap():
    member_gaps    = []
    nonmember_gaps = []
    valid_domains  = []

    for dom in DOMAINS:
        ev = load_pt(os.path.join(RUN3_BASE, dom, "verify_elbo.pt"), silent=True)
        if ev is None: continue
        member_gaps.append(float(ev["member_gap"]))
        nonmember_gaps.append(float(ev["nonmember_gap"]))
        valid_domains.append(DOMAIN_LABELS[dom])

    x = np.arange(len(valid_domains))
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    b1 = ax.bar(x - width / 2, member_gaps,    width, label="Member",
                color="#5c4b8a", alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + width / 2, nonmember_gaps, width, label="Non-member",
                color="#e05c5c", alpha=0.88, edgecolor="white")

    ax.axhline(0, color="black", lw=0.5)
    max_val = max(member_gaps + nonmember_gaps)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + max_val * 0.015,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_domains, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("ELBO Gap (nats)", fontsize=8)
    ax.set_title("Memorisation Verification: ELBO Gap per Domain", fontsize=9, fontweight="bold")
    # legend top-right, clear of bars
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max_val * 1.22)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_elbo_gap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  fig7_elbo_gap.pdf")


# ---------------------------------------------------------------------------
# Appendix Fig A — ROC grid (2×3)
# ---------------------------------------------------------------------------
def composite_appendix_roc():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    fig.suptitle("Per-Domain ROC Curves", fontsize=9, fontweight="bold")
    methods_cfg = [
        ("Loss",    PALETTE["Loss"],    "-"),
        ("Zlib",    PALETTE["Zlib"],    "--"),
        ("SAMA",    PALETTE["SAMA"],    "-."),
        ("XGBoost", PALETTE["XGBoost"], "-"),
        ("MLP",     PALETTE["MLP"],     "--"),
    ]

    for idx, (dom, ax) in enumerate(zip(DOMAINS, axes.flat)):
        cls, sama, loss, zlib, ratio = load_run3_domain(dom)
        if cls is None: ax.set_visible(False); continue
        y = cls["y_true"]
        ax.plot([0, 1], [0, 1], "k:", lw=0.6, alpha=0.4)
        for name, color, ls in methods_cfg:
            sc = {"Loss": loss["scores"] if loss else None,
                  "Zlib": zlib["scores"] if zlib else None,
                  "SAMA": sama["scores"] if sama else None,
                  "XGBoost": cls["xgb_probs"],
                  "MLP": cls["mlp_probs"]}[name]
            if sc is None: continue
            fpr, tpr, a = compute_roc(sc, y)
            ax.plot(fpr, tpr, color=color, ls=ls, lw=1.2, label=f"{name} ({a:.3f})")
        ax.set_title(DOMAIN_LABELS[dom], fontsize=8, fontweight="bold")
        ax.set_xlabel("FPR", fontsize=7)
        ax.set_ylabel("TPR", fontsize=7)
        ax.tick_params(labelsize=6.5)
        if idx == 0:
            ax.legend(fontsize=5.5, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "appx_roc_grid.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  appx_roc_grid.pdf")


# ---------------------------------------------------------------------------
# Appendix Fig B — PR grid (2×3)
# ---------------------------------------------------------------------------
def composite_appendix_pr():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    fig.suptitle("Per-Domain Precision-Recall Curves", fontsize=9, fontweight="bold")
    methods_cfg = [
        ("Loss",    PALETTE["Loss"],    "-"),
        ("Zlib",    PALETTE["Zlib"],    "--"),
        ("SAMA",    PALETTE["SAMA"],    "-."),
        ("XGBoost", PALETTE["XGBoost"], "-"),
        ("MLP",     PALETTE["MLP"],     "--"),
    ]

    for idx, (dom, ax) in enumerate(zip(DOMAINS, axes.flat)):
        cls, sama, loss, zlib, ratio = load_run3_domain(dom)
        if cls is None: ax.set_visible(False); continue
        y = cls["y_true"]
        baseline = float(y.float().mean())
        ax.axhline(baseline, color="gray", lw=0.6, ls=":", alpha=0.5)
        for name, color, ls in methods_cfg:
            sc = {"Loss": loss["scores"] if loss else None,
                  "Zlib": zlib["scores"] if zlib else None,
                  "SAMA": sama["scores"] if sama else None,
                  "XGBoost": cls["xgb_probs"],
                  "MLP": cls["mlp_probs"]}[name]
            if sc is None: continue
            rec, prec, ap = compute_pr(sc, y)
            ax.plot(rec, prec, color=color, ls=ls, lw=1.2, label=f"{name} (AP={ap:.3f})")
        ax.set_title(DOMAIN_LABELS[dom], fontsize=8, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=7)
        ax.set_ylabel("Precision", fontsize=7)
        ax.tick_params(labelsize=6.5)
        if idx == 0:
            ax.legend(fontsize=5.5, loc="upper right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "appx_pr_grid.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  appx_pr_grid.pdf")


# ---------------------------------------------------------------------------
# Appendix Fig C — Shadow conditions heatmap
# ---------------------------------------------------------------------------
def plot_appx_shadow_conditions():
    cond_keys   = ["A_oracle", "B_full46", "C_elbo_entropy8", "D_attn16", "E_pruned30"]
    cond_labels = ["Oracle (A)", "Shadow-46 (B)", "ELBO+H (C)", "Attn-only (D)", "Pruned-30 (E)"]
    dom_labels  = [DOMAIN_LABELS[d] for d in DOMAINS]
    data        = np.zeros((len(cond_keys), len(DOMAINS)))

    for j, dom in enumerate(DOMAINS):
        tr  = load_pt(os.path.join(RUN4_BASE, dom, "transfer_results.pt"), silent=True)
        cls = load_pt(os.path.join(RUN3_BASE, dom, "classifier_results.pt"), silent=True)
        if tr is None or cls is None: continue
        y = cls["y_true"]
        for i, ck in enumerate(cond_keys):
            cond = tr["conditions"].get(ck)
            if cond and "probs" in cond:
                _, _, a = compute_roc(torch.tensor(cond["probs"]), y)
                data[i, j] = a

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    sns.heatmap(data, annot=True, fmt=".3f",
                xticklabels=dom_labels, yticklabels=cond_labels,
                cmap="YlOrRd", vmin=0.5, vmax=1.0,
                linewidths=0.4, linecolor="#cccccc",
                ax=ax, cbar_kws={"label": "AUC-ROC", "shrink": 0.9},
                annot_kws={"size": 7})
    ax.set_title("Shadow Model: AUC by Condition and Domain", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0,  labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "appx_shadow_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  appx_shadow_heatmap.pdf")


# ---------------------------------------------------------------------------
# Appendix Fig D — Feature KDE plots (top 4 groups, arxiv)
# ---------------------------------------------------------------------------
def plot_appx_feature_kde():
    X    = load_pt(os.path.join(RUN3_BASE, "arxiv", "X.pt"), silent=True)
    y    = load_pt(os.path.join(RUN3_BASE, "arxiv", "classifier_results.pt"), silent=True)
    sigs = load_pt(os.path.join(RUN3_BASE, "arxiv", "signal_names.pt"), silent=True)
    if X is None or y is None or sigs is None:
        print("  [SKIP] appx_feature_kde.pdf — missing data")
        return

    labels  = y["y_true"].numpy()
    X_np    = X.numpy()
    mem_idx = labels == 1
    nom_idx = labels == 0

    # Top-4 feature indices: elbo_t0, elbo_t1, elbo_t2, pred_entropy_t0
    top_sigs = ["elbo_t0_a0.05", "elbo_t1_a0.20", "elbo_t2_a0.35", "pred_entropy_t0"]
    top_idx  = [sigs.index(s) for s in top_sigs if s in sigs]

    fig, axes = plt.subplots(1, len(top_idx), figsize=(7.0, 2.0), sharey=False)
    fig.suptitle("KDE of Top Features: Member vs. Non-member (arXiv)", fontsize=8.5, fontweight="bold")

    nice_labels = {
        "elbo_t0_a0.05": r"ELBO @ $\alpha$=0.05",
        "elbo_t1_a0.20": r"ELBO @ $\alpha$=0.20",
        "elbo_t2_a0.35": r"ELBO @ $\alpha$=0.35",
        "pred_entropy_t0": r"Entropy @ $\alpha$=0.05",
    }

    for ax, (sig, idx) in zip(axes, zip(top_sigs, top_idx)):
        feat = X_np[:, idx]
        sns.kdeplot(feat[mem_idx], ax=ax, color=PALETTE["XGBoost"], label="Member",
                    fill=True, alpha=0.35, linewidth=1.2)
        sns.kdeplot(feat[nom_idx], ax=ax, color=PALETTE["SAMA"], label="Non-member",
                    fill=True, alpha=0.35, linewidth=1.2, linestyle="--")
        ax.set_title(nice_labels.get(sig, sig), fontsize=7.5)
        ax.set_xlabel("Value", fontsize=7)
        ax.tick_params(labelsize=6.5)
        if ax == axes[0]:
            ax.set_ylabel("Density", fontsize=7)
            ax.legend(fontsize=6.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "appx_feature_kde.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  appx_feature_kde.pdf")


# ---------------------------------------------------------------------------
# Appendix Fig E — Run 1 POC: ELBO trajectory curves (member vs non-member)
# ---------------------------------------------------------------------------
def plot_appx_run1_elbo():
    ev = load_pt(os.path.join(RUN1_BASE, "verify_elbo.pt"), silent=True)
    if ev is None:
        # Try constructing from X.pt + y.pt
        X  = load_pt(os.path.join(RUN1_BASE, "X.pt"), silent=True)
        y  = load_pt(os.path.join(RUN1_BASE, "y.pt"), silent=True)
        if X is None or y is None:
            print("  [SKIP] appx_run1_elbo.pdf — missing data")
            return
        # First 4 dims are ELBO at t0..t3
        X_np = X.numpy(); y_np = y.numpy()
        mem  = X_np[y_np == 1, :4]
        nom  = X_np[y_np == 0, :4]
    else:
        # Use box plot of gap distributions
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        data = [ev["base_member_elbo"].numpy() - ev["ft_member_elbo"].numpy(),
                ev["base_nonmember_elbo"].numpy() - ev["ft_nonmember_elbo"].numpy()]
        bp = ax.boxplot(data, labels=["Member", "Non-member"], patch_artist=True,
                        widths=0.5, medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor(PALETTE["XGBoost"])
        bp["boxes"][1].set_facecolor(PALETTE["SAMA"])
        for patch in bp["boxes"]:
            patch.set_alpha(0.75)
        ax.set_ylabel(r"ELBO Gap $L_{\mathrm{base}} - L_{\mathrm{FT}}$ (nats)", fontsize=8)
        ax.set_title("Run 1 POC: ELBO Gap Distribution (GitHub, 15-epoch)", fontsize=8.5, fontweight="bold")
        ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "appx_run1_elbo.pdf"), bbox_inches="tight")
        plt.close(fig)
        print("  appx_run1_elbo.pdf")
        return

    t = [0.05, 0.20, 0.35, 0.50]
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(t, mem.mean(0),    "o-",  color=PALETTE["XGBoost"], lw=1.8,
            label=f"Member (mean)", markersize=4)
    ax.fill_between(t, mem.mean(0) - mem.std(0), mem.mean(0) + mem.std(0),
                    alpha=0.15, color=PALETTE["XGBoost"])
    ax.plot(t, nom.mean(0),    "s--", color=PALETTE["SAMA"], lw=1.8,
            label=f"Non-member (mean)", markersize=4)
    ax.fill_between(t, nom.mean(0) - nom.std(0), nom.mean(0) + nom.std(0),
                    alpha=0.15, color=PALETTE["SAMA"])
    ax.set_xlabel(r"Masking ratio $\alpha$", fontsize=8)
    ax.set_ylabel("ELBO (nats)", fontsize=8)
    ax.set_xticks(t)
    ax.set_title("Run 1 POC: ELBO Trajectory (GitHub, 15-epoch)", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "appx_run1_elbo.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  appx_run1_elbo.pdf")


# ---------------------------------------------------------------------------
# Appendix Fig F — Group transfer AUC (shadow model, per group)
# ---------------------------------------------------------------------------
def plot_appx_group_transfer():
    group_aucs = {}
    for dom in DOMAINS:
        tr = load_pt(os.path.join(RUN4_BASE, dom, "transfer_results.pt"), silent=True)
        if tr is None: continue
        for grp, auc_val in tr.get("group_transfer", {}).items():
            group_aucs.setdefault(grp, []).append(float(auc_val))

    if not group_aucs:
        print("  [SKIP] appx_group_transfer.pdf — no data")
        return

    sorted_groups = sorted(group_aucs, key=lambda g: np.mean(group_aucs[g]), reverse=True)
    means  = [np.mean(group_aucs[g]) for g in sorted_groups]
    stds   = [np.std(group_aucs[g])  for g in sorted_groups]
    labels = [FEATURE_LABELS.get(g, g) for g in sorted_groups]

    cmap = plt.cm.coolwarm_r
    norm_vals = (np.array(means) - 0.5) / (max(means) - 0.5 + 1e-9)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bars = ax.barh(range(len(labels)), means, xerr=stds,
                   color=[cmap(v) for v in norm_vals],
                   edgecolor="#555", lw=0.4, capsize=2, error_kw={"lw": 0.8})
    ax.axvline(0.5, color="red", lw=0.8, ls="--", alpha=0.7, label="Random (0.5)")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Transfer AUC (mean ± std, 6 domains)", fontsize=8)
    ax.set_title("Shadow Model: Per-Group Transfer AUC", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7)
    for bar, val in zip(bars, means):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "appx_group_transfer.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  appx_group_transfer.pdf")


# ---------------------------------------------------------------------------
# Appendix Fig G — TPR @ 1% FPR bar chart
# ---------------------------------------------------------------------------
def plot_appx_tpr1pct():
    methods = ["Loss", "Zlib", "SAMA", "XGBoost", "MLP"]
    colors  = [PALETTE["Loss"], PALETTE["Zlib"], PALETTE["SAMA"],
               PALETTE["XGBoost"], PALETTE["MLP"]]
    dom_labels = [DOMAIN_LABELS[d] for d in DOMAINS]
    data = {m: [] for m in methods}

    for dom in DOMAINS:
        cls, sama, loss, zlib, ratio = load_run3_domain(dom)
        if cls is None:
            for m in methods: data[m].append(0)
            continue
        y = cls["y_true"]
        for name, sc in [
            ("Loss",    loss["scores"] if loss else None),
            ("Zlib",    zlib["scores"] if zlib else None),
            ("SAMA",    sama["scores"] if sama else None),
            ("XGBoost", cls["xgb_probs"]),
            ("MLP",     cls["mlp_probs"]),
        ]:
            if sc is not None:
                fpr, tpr, _ = compute_roc(sc, y)
                data[name].append(tpr_at_fpr(fpr, tpr, 0.01))
            else:
                data[name].append(0)

    x = np.arange(len(DOMAINS))
    width = 0.14
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for k, (name, color) in enumerate(zip(methods, colors)):
        offset = (k - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, [v * 100 for v in data[name]],
               width, label=name, color=color, alpha=0.87, edgecolor="white", lw=0.4)

    ax.set_xlabel("Domain", fontsize=8)
    ax.set_ylabel("TPR @ 1% FPR (%)", fontsize=8)
    ax.set_title("True Positive Rate at 1% FPR by Domain and Method", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dom_labels, rotation=22, ha="right", fontsize=7.5)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "appx_tpr1pct_bar.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  appx_tpr1pct_bar.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Generating plots → {OUT}")
    plot_fig1_concept()
    plot_fig2_auc_heatmap()
    plot_fig3_roc_aggregate()
    plot_fig4_tpr_bar()
    plot_fig5_ablation()
    plot_fig6_shadow()
    plot_fig7_elbo_gap()
    composite_appendix_roc()
    composite_appendix_pr()
    plot_appx_shadow_conditions()
    plot_appx_feature_kde()
    plot_appx_run1_elbo()
    plot_appx_group_transfer()
    plot_appx_tpr1pct()
    print(f"\nDone. {len(os.listdir(OUT))} files in {OUT}")
