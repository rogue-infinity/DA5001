"""
benchmark.py — Final comparison table: SAMA vs XGBoost vs MLP.

Inputs:
  results/sama_scores.pt         — {scores, labels}
  results/classifier_results.pt  — {xgb_probs, mlp_probs, y_true, metrics_xgb, metrics_mlp}

Outputs:
  Prints table to stdout.
  Logs wandb.Table to project="da5001-mia", run="benchmark".
"""

import argparse
import os

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(y_true, y_score, fpr_threshold: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(fpr_threshold, fpr, tpr))


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 fpr_thresholds: list[float], n_bootstraps: int = 1000,
                 seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    N = len(y_true)
    auc_s, tpr_s = [], {f: [] for f in fpr_thresholds}
    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, N)
        yb, sb = y_true[idx], y_score[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc_s.append(roc_auc_score(yb, sb))
        for f in fpr_thresholds:
            tpr_s[f].append(tpr_at_fpr(yb, sb, f))

    def ci(s):
        s = np.array(s) if s else np.array([0.0])
        return s.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)

    out = {}
    out["auc"], out["auc_lo"], out["auc_hi"] = ci(auc_s)
    for f in fpr_thresholds:
        m, lo, hi = ci(tpr_s[f])
        out[f"tpr_{f}"] = m
        out[f"tpr_{f}_lo"] = lo
        out[f"tpr_{f}_hi"] = hi
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_bootstraps", type=int, default=1000)
    args = parser.parse_args()

    wandb.init(project="da5001-mia", name="run2-benchmark", config=vars(args))

    # Load classifier results (defines y_true)
    clf = torch.load(os.path.join(args.results_dir, "classifier_results.pt"), weights_only=True)
    xgb_probs  = clf["xgb_probs"].numpy().astype(np.float64)
    mlp_probs  = clf["mlp_probs"].numpy().astype(np.float64)
    y_true     = clf["y_true"].numpy()
    n          = len(y_true)

    # Load all attack score files — align to n samples
    def _load_scores(fname):
        d = torch.load(os.path.join(args.results_dir, fname), weights_only=True)
        return d["scores"].numpy().astype(np.float64)[:n]

    FPR_THRESHOLDS = [0.001, 0.01, 0.10]

    methods = {}

    # Baseline attacks (loss, zlib, ratio, SAMA) — loaded if present
    for attack_name, display_name in [
        ("loss_scores",  "Loss (baseline)"),
        ("zlib_scores",  "Zlib (baseline)"),
        ("ratio_scores", "Ratio (baseline)"),
        ("sama_scores",  "SAMA (baseline)"),
    ]:
        fpath = os.path.join(args.results_dir, f"{attack_name}.pt")
        if os.path.exists(fpath):
            methods[display_name] = (_load_scores(f"{attack_name}.pt"), y_true)
        else:
            print(f"WARNING: {fpath} not found — skipping {display_name}")

    # Our classifiers
    methods["XGBoost (ours)"] = (xgb_probs, y_true)
    methods["MLP (ours)"]     = (mlp_probs, y_true)

    rows = []
    for name, (scores, labels) in methods.items():
        m = bootstrap_ci(labels, scores, FPR_THRESHOLDS, args.n_bootstraps)

        def fmt_ci(val, lo, hi):
            return f"{val*100:.1f}±{((hi-lo)/2)*100:.1f}%"

        row = {
            "Method": name,
            "AUC": f"{m['auc']:.4f} [{m['auc_lo']:.4f},{m['auc_hi']:.4f}]",
            "TPR@0.1%FPR": fmt_ci(m["tpr_0.001"], m["tpr_0.001_lo"], m["tpr_0.001_hi"]),
            "TPR@1%FPR": fmt_ci(m["tpr_0.01"], m["tpr_0.01_lo"], m["tpr_0.01_hi"]),
            "TPR@10%FPR": fmt_ci(m["tpr_0.1"], m["tpr_0.1_lo"], m["tpr_0.1_hi"]),
        }
        rows.append(row)

    # Print table
    col_widths = {k: max(len(k), max(len(r[k]) for r in rows)) for k in rows[0]}
    header = "  ".join(k.ljust(col_widths[k]) for k in col_widths)
    sep = "  ".join("─" * col_widths[k] for k in col_widths)
    print("\n" + header)
    print(sep)
    for row in rows:
        print("  ".join(row[k].ljust(col_widths[k]) for k in col_widths))
    print()

    # wandb table
    table = wandb.Table(columns=list(rows[0].keys()))
    for row in rows:
        table.add_data(*row.values())
    wandb.log({"benchmark/results": table})

    # Also log scalars for easy comparison
    for row in rows:
        name_key = row["Method"].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        for method_name, (scores, labels) in methods.items():
            if method_name == row["Method"]:
                m = bootstrap_ci(labels, scores, FPR_THRESHOLDS, n_bootstraps=100)
                wandb.log({
                    f"benchmark/{name_key}_auc": m["auc"],
                    f"benchmark/{name_key}_tpr1fpr": m["tpr_0.01"],
                })

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
