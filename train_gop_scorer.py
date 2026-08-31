"""Train XGBoost scorers on GOP features (phone/word/sentence) + calibration ablation.

Runs under the 3.14 env (xgboost 3.2.0 installed).
  --use_calib  include per-phone native-z (LibriSpeech) calibrated features
Models saved as xgboost native (ubj) for web deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

import gop_features as gf
from gop_data import load_split_ids

DATA_DIR = Path("gop_data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=DATA_DIR / "ctc_head_libri.pt")
    ap.add_argument("--use_calib", action="store_true",
                    help="include native-z (LibriSpeech) calibrated features")
    ap.add_argument("--train_ids_file", type=Path, default=None,
                    help="official train utt ids (one per line); else uses split.json")
    ap.add_argument("--test_ids_file", type=Path, default=None,
                    help="official test utt ids (one per line)")
    ap.add_argument("--tag", type=str, default=None,
                    help="output model tag (default calib/raw)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inv = json.loads((DATA_DIR / "inventory.json").read_text(encoding="utf-8"))
    V = len(inv)

    print(f"head: {args.head}  calib: {args.use_calib}")
    model = gf.load_head(args.head, V, device)
    datasets, split, _ = gf.build_datasets(model, V, device, args.use_calib)
    if args.train_ids_file and args.test_ids_file:
        tr = set(load_split_ids(args.train_ids_file))
        te = set(load_split_ids(args.test_ids_file))
        print(f"split override: train {len(tr)} / test {len(te)} utts")
        tag = args.tag or "official"
    else:
        tr = set(split["train"])
        te = set(split["test"])
        tag = args.tag or ("calib" if args.use_calib else "raw")
    tr = {u for u in tr}
    te = {u for u in te}

    # ---- phone level (binary correct) ----
    X, y, utts = datasets["phone"]
    m = np.array([u in tr for u in utts])
    base_auc = roc_auc_score(y[~m], X[~m, 0])  # plain logprob-GOP baseline
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", n_jobs=8,
        eval_metric="logloss",
    )
    clf.fit(X[m], y[m])
    auc = roc_auc_score(y[~m], clf.predict_proba(X[~m])[:, 1])
    clf.save_model(str(DATA_DIR / f"scorer_phone_{tag}.ubj"))
    print(f"\n== phone (n_tr={int(m.sum())}, n_te={int((~m).sum())}) ==")
    print(f"  baseline logprob-GOP AUC: {base_auc:.3f}   XGBoost AUC: {auc:.3f}")

    # ---- word level ----
    X, y, utts = datasets["word"]
    m = np.array([u in tr for u in utts])
    base = spearmanr(X[~m, 0], y[~m]).statistic
    reg = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", n_jobs=8,
    )
    reg.fit(X[m], y[m])
    pred = reg.predict(X[~m])
    rho = spearmanr(pred, y[~m]).statistic
    pcc = pearsonr(pred, y[~m]).statistic
    reg.save_model(str(DATA_DIR / f"scorer_word_{tag}.ubj"))
    print(f"\n== word (n_tr={int(m.sum())}, n_te={int((~m).sum())}) ==")
    print(f"  baseline mean-logprob: Spearman={base:.3f}")
    print(f"  XGBoost: Spearman={rho:.3f}  PCC={pcc:.3f}")

    # ---- sentence level ----
    X, y_acc, y_flu, utts = datasets["sentence"]
    m = np.array([u in tr for u in utts])
    base_acc = spearmanr(X[~m, 0], np.asarray(y_acc)[~m]).statistic
    base_flu = spearmanr(X[~m, 0], np.asarray(y_flu)[~m]).statistic
    reg = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", n_jobs=8,
    )
    reg.fit(X[m], np.asarray(y_acc)[m])
    pred = reg.predict(X[~m])
    rho_acc = spearmanr(pred, np.asarray(y_acc)[~m]).statistic
    pcc_acc = pearsonr(pred, np.asarray(y_acc)[~m]).statistic
    rho_flu = spearmanr(pred, np.asarray(y_flu)[~m]).statistic
    reg.save_model(str(DATA_DIR / f"scorer_sent_{tag}.ubj"))
    print(f"\n== sentence (n_tr={int(m.sum())}, n_te={int((~m).sum())}) ==")
    print(f"  baseline mean-logprob: vs acc Spearman={base_acc:.3f}  vs flu {base_flu:.3f}")
    print(f"  XGBoost: vs acc Spearman={rho_acc:.3f} PCC={pcc_acc:.3f}  vs flu {rho_flu:.3f}")

    # ---- sentence fluency regressor (separate target, used by GOPScorer) ----
    reg_flu = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", n_jobs=8,
    )
    reg_flu.fit(X[m], np.asarray(y_flu)[m])
    pred_flu = reg_flu.predict(X[~m])
    rho_flu2 = spearmanr(pred_flu, np.asarray(y_flu)[~m]).statistic
    pcc_flu = pearsonr(pred_flu, np.asarray(y_flu)[~m]).statistic
    reg_flu.save_model(str(DATA_DIR / f"scorer_sentflu_{tag}.ubj"))
    print(f"  fluency model: vs flu Spearman={rho_flu2:.3f} PCC={pcc_flu:.3f}")


if __name__ == "__main__":
    main()
