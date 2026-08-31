"""5-fold speaker-disjoint CV + simple XGBoost hyperparameter sweep (SO762).

Head fixed (ctc_head_libri.pt). Features built once, re-split per fold.
Grid: n_estimators x200/400, max_depth x4/6, lr x0.05/0.1 (sentence-level primary).
Reports mean +/- std over folds for phone AUC / word Spearman / sentence Spearman.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import gop_features as gf
from gop_data import speaker_of

DATA_DIR = Path("gop_data")

GRID = [(n, d, lr) for n in (200, 400) for d in (4, 6) for lr in (0.05, 0.1)]


def make_folds(utts: list[str], seed: int = 42, n_folds: int = 5) -> list[set[str]]:
    rng = np.random.default_rng(seed)
    spk = sorted({speaker_of(u) for u in utts})
    rng.shuffle(spk)
    folds = [set() for _ in range(n_folds)]
    for i, s in enumerate(spk):
        folds[i % n_folds].add(s)
    fold_utts = [set(u for u in utts if speaker_of(u) in folds[i]) for i in range(n_folds)]
    return fold_utts


def run_cv(X, y, utts, folds, level, cfg, metric):
    """Returns per-fold metric values."""
    out = []
    for f in range(len(folds)):
        tr = set().union(*[folds[j] for j in range(len(folds)) if j != f])
        te = folds[f]
        m_tr = np.array([u in tr for u in utts])
        m_te = np.array([u in te for u in utts])
        if level == "phone":
            m = xgb.XGBClassifier(
                n_estimators=cfg[0], max_depth=cfg[1], learning_rate=cfg[2],
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                n_jobs=8, eval_metric="logloss",
            )
            m.fit(X[m_tr], y[m_tr])
            out.append(metric(y[m_te], m.predict_proba(X[m_te])[:, 1]))
        else:
            m = xgb.XGBRegressor(
                n_estimators=cfg[0], max_depth=cfg[1], learning_rate=cfg[2],
                subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8,
            )
            m.fit(X[m_tr], y[m_tr])
            out.append(metric(y[m_te], m.predict(X[m_te])))
    return out


def spearman(y, pred):
    return spearmanr(pred, y).statistic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=DATA_DIR / "ctc_head_libri.pt")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inv = json.loads((DATA_DIR / "inventory.json").read_text(encoding="utf-8"))
    V = len(inv)
    print(f"head: {args.head}  folds: {args.folds}")
    model = gf.load_head(args.head, V, device)

    print("building features (once)...")
    datasets, _, _ = gf.build_datasets(model, V, device, use_calib=True)
    n_folds = args.folds

    # sentence-level sweep
    X, y_acc, y_flu, utts = datasets["sentence"]
    folds = make_folds(utts, n_folds=n_folds)
    print("\n== sentence sweep (5-fold mean Spearman vs accuracy) ==")
    grid_res = []
    for cfg in GRID:
        vals = run_cv(X, np.asarray(y_acc), utts, folds, "sent", cfg, spearman)
        grid_res.append((cfg, float(np.mean(vals)), float(np.std(vals))))
        print(f"  n={cfg[0]:3d} d={cfg[1]} lr={cfg[2]:.2f}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")
    best_cfg, best_mean, _ = max(grid_res, key=lambda t: t[1])
    print(f"best: n={best_cfg[0]} d={best_cfg[1]} lr={best_cfg[2]:.2f}  mean={best_mean:.3f}")

    # baseline + final metrics with best cfg
    print("\n== final 5-fold with best config ==")
    # baseline = mean of feature col 0 (agg mean logp_mean)
    bl = []
    for f in range(n_folds):
        te = folds[f]
        m_te = np.array([u in te for u in utts])
        bl.append(spearman(np.asarray(y_acc)[m_te], X[m_te, 0]))
    print(f"  baseline mean-logprob vs acc: {np.mean(bl):.3f} +- {np.std(bl):.3f}")
    v = run_cv(X, np.asarray(y_acc), utts, folds, "sent", best_cfg, spearman)
    print(f"  XGBoost vs acc:        {np.mean(v):.3f} +- {np.std(v):.3f}")
    v = run_cv(X, np.asarray(y_flu), utts, folds, "sent", best_cfg, spearman)
    print(f"  XGBoost vs fluency:    {np.mean(v):.3f} +- {np.std(v):.3f}")

    # word level
    X, y, utts = datasets["word"]
    folds = make_folds(utts, n_folds=n_folds)
    bl = []
    for f in range(n_folds):
        m_te = np.array([u in folds[f] for u in utts])
        bl.append(spearman(y[m_te], X[m_te, 0]))
    print(f"\n  word baseline: {np.mean(bl):.3f} +- {np.std(bl):.3f}")
    v = run_cv(X, y, utts, folds, "word", best_cfg, spearman)
    print(f"  word XGBoost:  {np.mean(v):.3f} +- {np.std(v):.3f}")

    # phone level
    X, y, utts = datasets["phone"]
    folds = make_folds(utts, n_folds=n_folds)
    bl = []
    for f in range(n_folds):
        m_te = np.array([u in folds[f] for u in utts])
        bl.append(roc_auc_score(y[m_te], X[m_te, 0]))
    print(f"\n  phone baseline AUC: {np.mean(bl):.3f} +- {np.std(bl):.3f}")
    v = run_cv(X, y, utts, folds, "phone", best_cfg, roc_auc_score)
    print(f"  phone XGBoost AUC:   {np.mean(v):.3f} +- {np.std(v):.3f}")

    # save best-config models trained on all data
    tag = f"cv_best_n{best_cfg[0]}_d{best_cfg[1]}_lr{best_cfg[2]:.2f}"
    X_s, y_acc_all, y_flu_all, _ = datasets["sentence"]
    for name, Xm, ym, kind in (
        ("phone", datasets["phone"][0], datasets["phone"][1], "cls"),
        ("word", datasets["word"][0], datasets["word"][1], "reg"),
        ("sent", X_s, np.asarray(y_acc_all), "reg"),
    ):
        if kind == "cls":
            m = xgb.XGBClassifier(
                n_estimators=best_cfg[0], max_depth=best_cfg[1], learning_rate=best_cfg[2],
                subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8,
                eval_metric="logloss")
        else:
            m = xgb.XGBRegressor(
                n_estimators=best_cfg[0], max_depth=best_cfg[1], learning_rate=best_cfg[2],
                subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8)
        m.fit(Xm, ym)
        m.save_model(str(DATA_DIR / f"scorer_{name}_{tag}.ubj"))
    # sentence fluency (separate target)
    flu = xgb.XGBRegressor(
        n_estimators=best_cfg[0], max_depth=best_cfg[1], learning_rate=best_cfg[2],
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8)
    flu.fit(X_s, np.asarray(y_flu_all))
    flu.save_model(str(DATA_DIR / f"scorer_sentflu_{tag}.ubj"))
    print(f"\nsaved: scorer_*_{tag}.ubj (+ scorer_sentflu_{tag}.ubj)")


if __name__ == "__main__":
    main()
