"""Shared pieces: splits, feature matrices, models, tuning, metrics."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from telemetry_gen import FEATURES
from . import baseline as bl

# model inputs: every numeric feature, plus transport class split into three
# 0/1 columns (one-hot), because models need numbers, not category names
NUMERIC = [f for f in FEATURES if f != "transport_class"]
TRANSPORTS = ["mpls", "broadband", "cellular"]
X_COLS = NUMERIC + [f"transport_{t}" for t in TRANSPORTS]
# site-relative metrics that the re-baselining ablation rescales
RENORM_COLS = ["latency_ms", "jitter_ms", "loss_pct", "policy_sync_age_min"]
SPLIT = (0.50, 0.20, 0.30)            # train / val / test, by device
FAR_GRID = np.linspace(0, 1, 101)     # for averaging FAR/FRR trade-off curves


# ---- splits -----------------------------------------------------------------
def device_split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Device-grouped split, stratified on whether the device is ever compromised.
    No device contributes rows to more than one split."""
    rng = np.random.default_rng(seed)
    # one entry per device: 1 if the device is ever compromised, else 0
    dev = df.groupby("device_id").is_compromised.max()
    parts = {"train": [], "val": [], "test": []}
    # split compromised and clean devices separately (stratification), so every
    # split gets its fair share of the rare compromised devices
    for _, ids in dev.groupby(dev):
        ids = rng.permutation(ids.index.values)          # shuffle this group's device IDs
        n = len(ids)
        a, b = int(round(SPLIT[0] * n)), int(round((SPLIT[0] + SPLIT[1]) * n))   # 50% / 20% / 30% cut points
        parts["train"] += list(ids[:a]); parts["val"] += list(ids[a:b]); parts["test"] += list(ids[b:])
    # turn the device ID lists back into row-level DataFrames
    return {k: df[df.device_id.isin(v)].reset_index(drop=True) for k, v in parts.items()}


def rebaseline(df: pd.DataFrame, ref_medians: dict[str, float]) -> pd.DataFrame:
    """Ablation: rescale site-relative channel/sync metrics so the site's own
    median (label-free, all rows) matches the reference site's median."""
    out = df.copy()
    for c in RENORM_COLS:
        out[c] = out[c] * (ref_medians[c] / df[c].median())
    return out


def matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model input matrix: numeric features plus one-hot transport."""
    X = df[NUMERIC].astype(float).copy()
    for t in TRANSPORTS:
        X[f"transport_{t}"] = (df.transport_class == t).astype(float)   # 1.0 if this row uses transport t
    return X[X_COLS]                     # fixed column order, so every model sees the same layout


# ---- models -----------------------------------------------------------------
# hyperparameter settings tried for each model (every combination is fit once)
GRIDS = {
    "LogReg": [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],          # C = inverse regularization strength
    "RandomForest": [dict(max_depth=d, min_samples_leaf=l) for d, l in itertools.product((8, 16, None), (5, 20))],
    "GradBoost": [dict(learning_rate=lr, max_leaf_nodes=m) for lr, m in itertools.product((0.05, 0.1), (15, 31))],
}
MODELS = list(GRIDS)


def make_model(name: str, params: dict, seed: int):
    """Construct an untrained model of the given type with one hyperparameter setting. class_weight="balanced" up-weights the rare compromised class in every model."""
    if name == "LogReg":
        # logistic regression is sensitive to feature scale, so standardize first
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000, **params))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed, **params)
    if name == "GradBoost":
        # early stopping holds out 15% of training rows and stops adding trees
        # once 20 rounds pass without improvement
        return HistGradientBoostingClassifier(max_iter=300, early_stopping=True, validation_fraction=0.15,
                                              n_iter_no_change=20, l2_regularization=1.0,
                                              class_weight="balanced", random_state=seed, **params)
    raise ValueError(name)


def tune_and_fit(name, train, val, seed):
    """Grid search on the SOURCE site's validation split (PR-AUC); refit is the chosen model (no target-site data is ever used for selection)."""
    Xtr, ytr, Xva, yva = matrix(train), train.is_compromised, matrix(val), val.is_compromised
    best = (-1, None, None)                     # (validation PR-AUC, params, fitted model)
    # train one model per grid setting and keep whichever scores best on validation
    for p in GRIDS[name]:
        m = make_model(name, p, seed).fit(Xtr, ytr)
        s = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        if s > best[0]:
            best = (s, p, m)
    return best[2], best[1], best[0]            # model, chosen params, its validation PR-AUC


def score(name, model, df):
    """Risk score, higher = more likely compromised."""
    if name.startswith("Baseline"):        # model = source calibration dict, or None for fixed anchors
        return bl.risk_score(df, calibration=model)
    return model.predict_proba(matrix(df))[:, 1]     # learned models: probability of "compromised"


# ---- metrics ----------------------------------------------------------------
def f1_threshold(y, s):
    """Threshold maximizing F1 on the given (validation) data."""
    p, r, t = precision_recall_curve(y, s)
    # F1 at every candidate threshold (tiny epsilon avoids dividing by zero);
    # the last precision/recall pair has no threshold, so it is dropped
    f1 = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0)[:-1]
    return float(t[np.argmax(f1)])


def evaluate(y, s, thr, df):
    """
    Row-level detection metrics at threshold thr, plus device-level recall.
    FAR = compromised rows granted access (score below thr).
    FRR = non-compromised rows denied (score at/above thr).
    """
    y = np.asarray(y); pred = s >= thr                     # pred = True means "flag / deny"
    # confusion-matrix counts for the compromised class
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    deg = (df.label == "degraded").values                  # benign-but-unhealthy rows
    # device-level recall: was each compromised device flagged on at least one row?
    flagged = pd.Series(pred & (y == 1)).groupby(df.device_id.values).any()
    comp_dev = df.groupby("device_id").is_compromised.max()
    return dict(
        pr_auc=average_precision_score(y, s), precision=prec, recall=rec,
        f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        far=1 - rec, frr=float(pred[y == 0].mean()),
        frr_degraded=float(pred[deg].mean()) if deg.any() else np.nan,
        frr_healthy=float(pred[(df.label == "healthy").values].mean()),
        device_recall=float(flagged.reindex(comp_dev[comp_dev == 1].index, fill_value=False).mean()),
        base_rate=float(y.mean()),                         # share of compromised rows (= chance PR-AUC)
    )


def far_frr_curve(y, s):
    """FRR interpolated on a fixed FAR grid (threshold-free trade-off curve)."""
    fpr, tpr, _ = roc_curve(y, s)          # sweep every threshold at once
    far = 1 - tpr                          # decreasing as threshold drops
    order = np.argsort(far)                # np.interp needs increasing x values
    # put every run on the same 101-point FAR grid so curves can be averaged across seeds
    return np.interp(FAR_GRID, far[order], fpr[order])


def per_decision_us(name, model, df, n_single=300, n_batch=20000):
    """Median single-decision latency and amortized batch latency, microseconds."""
    # single decisions: time 300 one-row calls (dominated by Python/pandas overhead)
    rows = df.sample(n_single, random_state=0)
    single = []
    for i in range(n_single):
        r = rows.iloc[[i]]
        t0 = time.perf_counter(); score(name, model, r); single.append(time.perf_counter() - t0)
    # batch: score 20,000 rows at once, 5 repetitions, and divide by the row count
    batch = df.sample(min(n_batch, len(df)), random_state=1)
    tb = []
    for _ in range(5):
        t0 = time.perf_counter(); score(name, model, batch); tb.append(time.perf_counter() - t0)
    return float(np.median(single) * 1e6), float(np.median(tb) / len(batch) * 1e6)
