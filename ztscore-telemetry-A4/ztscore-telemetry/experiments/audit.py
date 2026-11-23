"""Bias audit: checks for design choices that could push results in an artificial direction.

  1. source_anchored  — the baseline's absolute link anchors were written for a clean,
     Site-A-like network. Here they are re-anchored to the SOURCE site (each link/sync
     metric rescaled by median_A / median_source, label-free), so the baseline is
     "calibrated where it was built", exactly like the learned models.
  2. no_transport     — healthy devices never leave their primary transport in the
     generator, which may hand learned models a shortcut. Retrain without transport_class.

    python -m experiments.audit --seeds 0 1 2 3 4 --out results/audit
"""
from __future__ import annotations

import argparse, pathlib, sys, warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# make the project root importable when this file is run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from telemetry_gen import DEFAULT_CONFIG, generate      # noqa: E402
from experiments import common as C                     # noqa: E402
from experiments import baseline as bl                  # noqa: E402

warnings.filterwarnings("ignore")


def source_anchored(df, src_medians, ref_medians):
    """Rescale link/sync metrics by reference median / source median, so the baseline's fixed anchors are interpreted relative to the site it was built on."""
    out = df.copy()
    for c in C.RENORM_COLS:
        out[c] = out[c] * (ref_medians[c] / src_medians[c])
    return out


def main(seeds, out, models=("LogReg", "GradBoost")):
    """Run both audit checks in all four train->test conditions and save PR-AUC per run."""
    rows = []
    for seed in seeds:
        data = generate(DEFAULT_CONFIG, seed)
        sp = {k: C.device_split(v, seed) for k, v in data.items()}
        ref = {c: data["A"][c].median() for c in C.RENORM_COLS}       # reference = this seed's Site A medians
        for src, tgt in (("A", "B"), ("B", "A")):
            # check 1: baseline anchored to its source site
            srcmed = {c: data[src][c].median() for c in C.RENORM_COLS}
            for test in (src, tgt):
                te = sp[test]["test"]
                y = te.is_compromised
                rows.append(dict(seed=seed, check="source_anchored", model="Baseline",
                                 condition=f"{src}->{test}",
                                 pr_auc=average_precision_score(y, bl.risk_score(source_anchored(te, srcmed, ref)))))
            # check 2: learned models without transport_class
            keep = [c for c in C.X_COLS if not c.startswith("transport_")]
            for name in models:
                # temporarily shrink the shared feature list, then always restore it
                # (the try/finally guarantees the restore even if training fails)
                orig = C.X_COLS[:]
                C.X_COLS[:] = keep
                try:
                    m = C.tune_and_fit(name, sp[src]["train"], sp[src]["val"], seed)[0]
                    for test in (src, tgt):
                        te = sp[test]["test"]
                        rows.append(dict(seed=seed, check="no_transport", model=name, condition=f"{src}->{test}",
                                         pr_auc=average_precision_score(te.is_compromised,
                                                                        m.predict_proba(C.matrix(te))[:, 1])))
                finally:
                    C.X_COLS[:] = orig
        print(f"seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "audit.csv", index=False)
    # summary: mean and SD of PR-AUC for each check, method and condition
    print(df.groupby(["check", "model", "condition"]).pr_auc.agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/audit")
    a = ap.parse_args(); main(a.seeds, a.out)
