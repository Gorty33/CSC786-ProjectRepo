"""Recall by kill-chain stage at each method's source-val operating threshold.

Shows which parts of the UAT-8616 kill chain each method catches, and how much
of the false-accept rate is the low-signal `dwell` stage.

    python -m experiments.stage_recall --seeds 0 1 2 3 4 --out results/paper
"""
from __future__ import annotations

import argparse, pathlib, sys, warnings

import pandas as pd

# make the project root importable when this file is run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from telemetry_gen import DEFAULT_CONFIG, generate      # noqa: E402
from experiments import common as C                     # noqa: E402

warnings.filterwarnings("ignore")
# kill-chain stages in the order they happen (column order for the output table)
STAGES = ["initial_access", "rollback", "restore", "persistence", "config_manipulation", "dwell"]


def main(seeds, out):
    """Train every method on Site A, then measure what share of each stage's compromised rows it flags on the Site A and Site B test splits."""
    rows = []
    for seed in seeds:
        data = generate(DEFAULT_CONFIG, seed)
        sp = {k: C.device_split(v, seed) for k, v in data.items()}
        tr, va = sp["A"]["train"], sp["A"]["val"]
        # same setup as the main run: calibrated baseline + tuned learned models, all from Site A
        fitted = {"Baseline": C.bl.calibrate(tr)}
        for name in C.MODELS:
            fitted[name] = C.tune_and_fit(name, tr, va, seed)[0]
        for name, m in fitted.items():
            thr = C.f1_threshold(va.is_compromised, C.score(name, m, va))   # operating threshold from Site A val
            for test in "AB":
                te = sp[test]["test"]
                # keep only compromised rows and mark which ones the method flagged ("hit")
                comp = te[te.is_compromised == 1].assign(hit=C.score(name, m, te)[te.is_compromised.values == 1] >= thr)
                # recall within each kill-chain stage
                for st, g in comp.groupby("stage"):
                    rows.append(dict(seed=seed, model=name, condition=f"A->{test}", stage=st,
                                     n=len(g), recall=g.hit.mean()))
        print(f"seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stage_recall_raw.csv", index=False)
    # average recall over seeds: one row per (condition, method), one column per stage
    piv = (df.groupby(["condition", "model", "stage"]).recall.mean().unstack("stage")
             .reindex(columns=STAGES).round(3))
    # how much of the compromised test data each stage makes up
    share = df.groupby(["condition", "stage"]).n.sum().unstack().reindex(columns=STAGES)
    share = (share.T / share.sum(axis=1)).T.round(3)
    piv.to_csv(out / "table7_stage_recall.csv")
    text = ["## Table 7 — Row recall by kill-chain stage (A-trained, source-val threshold)\n",
            piv.to_markdown(), "", "Share of compromised test rows by stage:\n", share.to_markdown(), ""]
    (out / "stage_recall.md").write_text("\n".join(text)); print("\n".join(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/paper")
    a = ap.parse_args(); main(a.seeds, a.out)
