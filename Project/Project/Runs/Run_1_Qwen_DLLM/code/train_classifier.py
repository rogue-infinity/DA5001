"""
train_classifier.py — Train XGBoost + MLP classifiers with 5-fold stratified CV.

Inputs:
  results/X.pt  — feature matrix [N, D]
  results/y.pt  — labels [N]

Outputs:
  results/classifier_results.pt — {xgb_probs, mlp_probs, y_true, metrics_xgb, metrics_mlp}
"""

import argparse
import os

import numpy as np
import torch
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False
    print("xgboost not found — falling back to sklearn GradientBoostingClassifier")


def tpr_at_fpr(y_true, y_score, fpr_threshold: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= fpr_threshold)[0]
    return float(tpr[idx[-1]]) if len(idx) > 0 else 0.0


def bootstrap_metrics(y_true: np.ndarray, y_score: np.ndarray,
                      fpr_thresholds: list[float], n_bootstraps: int = 1000,
                      seed: int = 42) -> dict:
    """Compute AUC + TPR@FPR with 95% bootstrap CIs."""
    rng = np.random.RandomState(seed)
    N = len(y_true)

    auc_samples = []
    tpr_samples = {f: [] for f in fpr_thresholds}

    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, N)
        y_b, s_b = y_true[idx], y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_samples.append(roc_auc_score(y_b, s_b))
        for f in fpr_thresholds:
            tpr_samples[f].append(tpr_at_fpr(y_b, s_b, f))

    def ci(samples):
        if not samples:
            return 0.0, 0.0, 0.0
        s = np.array(samples)
        return float(s.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

    result = {}
    result["auc_mean"], result["auc_lo"], result["auc_hi"] = ci(auc_samples)
    for f in fpr_thresholds:
        key = f"tpr_at_{f}"
        m, lo, hi = ci(tpr_samples[f])
        result[key] = m
        result[f"{key}_lo"] = lo
        result[f"{key}_hi"] = hi
    return result


def make_classifiers():
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, eval_metric="logloss",
            random_state=42,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        xgb = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=300,
        early_stopping=True,
        random_state=42,
    )
    return xgb, mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_bootstraps", type=int, default=1000)
    args = parser.parse_args()

    wandb.init(project="da5001-mia", name="classifiers", config=vars(args))

    X = torch.load(os.path.join(args.results_dir, "X.pt"), weights_only=True).numpy().astype(np.float32)
    y = torch.load(os.path.join(args.results_dir, "y.pt"), weights_only=True).numpy()
    print(f"Loaded X {X.shape}, y {y.shape}, class balance: {y.mean():.2%} members")

    FPR_THRESHOLDS = [0.001, 0.01, 0.10]
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    xgb_oof = np.zeros(len(y))
    mlp_oof = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{args.n_folds}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        xgb_clf, mlp_clf = make_classifiers()

        xgb_clf.fit(X_train_s, y_train)
        xgb_probs = xgb_clf.predict_proba(X_val_s)[:, 1]
        xgb_oof[val_idx] = xgb_probs

        mlp_clf.fit(X_train_s, y_train)
        mlp_probs = mlp_clf.predict_proba(X_val_s)[:, 1]
        mlp_oof[val_idx] = mlp_probs

        fold_xgb_auc = roc_auc_score(y_val, xgb_probs)
        fold_mlp_auc = roc_auc_score(y_val, mlp_probs)
        print(f"  XGB AUC: {fold_xgb_auc:.4f}  |  MLP AUC: {fold_mlp_auc:.4f}")
        wandb.log({
            f"classifiers/fold{fold+1}_xgb_auc": fold_xgb_auc,
            f"classifiers/fold{fold+1}_mlp_auc": fold_mlp_auc,
        })

    print("\nComputing overall OOF metrics with bootstrap CIs...")
    metrics_xgb = bootstrap_metrics(y, xgb_oof, FPR_THRESHOLDS, args.n_bootstraps)
    metrics_mlp = bootstrap_metrics(y, mlp_oof, FPR_THRESHOLDS, args.n_bootstraps)

    def fmt(m, key):
        lo = m.get(f"{key}_lo", 0)
        hi = m.get(f"{key}_hi", 0)
        return f"{m[key]:.4f} [{lo:.4f}, {hi:.4f}]"

    print(f"\nXGBoost OOF:")
    print(f"  AUC:         {fmt(metrics_xgb, 'auc_mean')}")
    for f in FPR_THRESHOLDS:
        print(f"  TPR@{f*100:.1f}%FPR: {fmt(metrics_xgb, f'tpr_at_{f}')}")

    print(f"\nMLP OOF:")
    print(f"  AUC:         {fmt(metrics_mlp, 'auc_mean')}")
    for f in FPR_THRESHOLDS:
        print(f"  TPR@{f*100:.1f}%FPR: {fmt(metrics_mlp, f'tpr_at_{f}')}")

    wandb.log({
        "classifiers/xgb_auc": metrics_xgb["auc_mean"],
        "classifiers/mlp_auc": metrics_mlp["auc_mean"],
        "classifiers/xgb_tpr_at_1fpr": metrics_xgb["tpr_at_0.01"],
        "classifiers/mlp_tpr_at_1fpr": metrics_mlp["tpr_at_0.01"],
    })

    os.makedirs(args.results_dir, exist_ok=True)
    torch.save({
        "xgb_probs": torch.tensor(xgb_oof),
        "mlp_probs": torch.tensor(mlp_oof),
        "y_true": torch.tensor(y),
        "metrics_xgb": metrics_xgb,
        "metrics_mlp": metrics_mlp,
        "fpr_thresholds": FPR_THRESHOLDS,
    }, os.path.join(args.results_dir, "classifier_results.pt"))
    print(f"\nSaved results/classifier_results.pt")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
