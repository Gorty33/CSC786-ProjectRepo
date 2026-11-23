"""Main experiment matrix.

    python -m experiments.run_experiments --seeds 0 1 2 3 4 --out results
    python -m experiments.run_experiments --seeds 0 1 2 3 4 --out results/ablation_rebaseline --rebaseline

For each seed: regenerate both sites, split each by device, then for each
source site S in {A, B} and target T:
  * fit + tune every learned model on S-train / S-val only
  * choose each method's operating threshold (max F1) on S-val only
  * evaluate on S-test (in-distribution) and T-test (cross-site)
The hand-weighted baseline has no fitted weights; it is calibrated on S-train
(label-free medians) and its operating threshold is chosen on S-val. It is also
reported at the prior paper's fixed management-plane threshold (theta = 0.85),
and with uncalibrated fixed anchors ("BaselineFixed") as a control.
"""
from __future__ import annotations

import argparse, json, pathlib, platform, sys, time, warnings

import numpy as np
import pandas as pd

# make the project root importable when this file is run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from telemetry_gen import DEFAULT_CONFIG, generate                    # noqa: E402
from experiments import common as C                                   # noqa: E402
from experiments import baseline as bl                                # noqa: E402

warnings.filterwarnings("ignore")


def shap_importance(name, model, df, seed, background, n=600):
    """
    Mean |SHAP| on n rows of df. Linear models are explained against the full
    training set (the reference distribution), not a subsample of the rows being
    explained. Tree models use path-dependent TreeSHAP, which needs no background.
    """
    import shap
    X = C.matrix(df.sample(min(n, len(df)), random_state=seed))     # rows to explain
    if name == "LogReg":
        # explain the logistic regression in its own standardized input space
        scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
        Xs, Xbg = scaler.transform(X), scaler.transform(C.matrix(background))
        masker = shap.maskers.Independent(Xbg, max_samples=len(Xbg))   # use every training row as background
        sv = shap.LinearExplainer(lr, masker).shap_values(Xs)
    else:
        sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
        # different SHAP versions return the positive class in different shapes;
        # normalize to one (rows x features) array for the "compromised" class
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    sv = pd.DataFrame(sv, columns=C.X_COLS)
    oh = [f"transport_{t}" for t in C.TRANSPORTS]
    sv["transport_class"] = sv[oh].sum(axis=1)     # group one-hot columns: signed sum per row, then |.|
    return sv.drop(columns=oh).abs().mean()        # mean |SHAP| per feature = its importance


def run(seeds, out, rebaseline=False, cfg=DEFAULT_CONFIG, do_shap=True, do_timing=True, models=C.MODELS, tag=""):
    """
    Run every method in all four train->test conditions for each seed, then
    write metrics, chosen hyperparameters, FAR/FRR curves, SHAP and timing to `out`.
    """
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    rows, curves, shap_rows, timing_rows, chosen = [], [], [], [], []
    for seed in seeds:
        t0 = time.time()
        data = generate(cfg, seed)                  # {"A": DataFrame, "B": DataFrame}, fresh for every seed
        if rebaseline:
            # ablation only: rescale every site's link metrics to Site A's medians
            ref = {c: data["A"][c].median() for c in C.RENORM_COLS}
            data = {k: C.rebaseline(v, ref) for k, v in data.items()}
        splits = {k: C.device_split(v, seed) for k, v in data.items()}

        # two directions: train on A and test on A and B, then train on B and test on B and A
        for src, tgt in (("A", "B"), ("B", "A")):
            tr, va = splits[src]["train"], splits[src]["val"]
            # Baseline: calibrated on the source training split (label-free medians);
            # BaselineFixed: the as-written absolute anchors, kept for comparison
            fitted = {"Baseline": bl.calibrate(tr), "BaselineFixed": None}
            # tune and train each learned model on the source site only
            for name in models:
                m, p, vs = C.tune_and_fit(name, tr, va, seed)
                fitted[name] = m
                chosen.append(dict(seed=seed, source=src, model=name, params=json.dumps(p), val_pr_auc=vs))

            # evaluate every method (baselines and learned models alike)
            for name, m in fitted.items():
                # operating threshold = max-F1 point on the SOURCE validation split
                thr = C.f1_threshold(va.is_compromised, C.score(name, m, va))
                variants = [(name, thr)]
                if name == "Baseline":
                    # also report the paper's fixed management-plane threshold (risk = 1 - 0.85)
                    variants.append(("Baseline@0.85", 1 - bl.THETA_MGMT))
                # test on the source site's test split (in-distribution), then the target's (cross-site)
                for test_site in (src, tgt):
                    te = splits[test_site]["test"]
                    s = C.score(name, m, te)
                    for vname, vthr in variants:
                        r = C.evaluate(te.is_compromised, s, vthr, te)
                        r.update(seed=seed, source=src, test=test_site, model=vname,
                                 condition=f"{src}->{test_site}",
                                 shift="in" if test_site == src else "cross", threshold=vthr)
                        rows.append(r)
                    curves.append(dict(seed=seed, source=src, test=test_site, model=name,
                                       frr=C.far_frr_curve(te.is_compromised, s).tolist()))
                # SHAP: learned models trained on Site A only, explained on Site A test rows
                if do_shap and not name.startswith("Baseline") and src == "A":
                    imp = shap_importance(name, m, splits["A"]["test"], seed, background=tr)
                    shap_rows += [dict(seed=seed, model=name, feature=f, mean_abs_shap=v) for f, v in imp.items()]
                # timing: measured once (first seed, Site A) per method
                if do_timing and seed == seeds[0] and src == "A" and name != "BaselineFixed":
                    single, batch = C.per_decision_us(name, m, splits["A"]["test"])
                    timing_rows.append(dict(model=name, single_us=single, batch_us_per_row=batch))
        print(f"{tag}seed {seed} done in {time.time() - t0:.0f}s", flush=True)

    # ---- save everything ----------------------------------------------------
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(chosen).to_csv(out / "chosen_hyperparams.csv", index=False)
    pd.DataFrame(curves).to_json(out / "far_frr_curves.json", orient="records")
    if shap_rows:
        pd.DataFrame(shap_rows).to_csv(out / "shap_importance.csv", index=False)
    if timing_rows:
        pd.DataFrame(timing_rows).to_csv(out / "timing.csv", index=False)
    # record seeds, settings and library versions alongside the results, for reproducibility
    import sklearn, scipy
    (out / "run_info.json").write_text(json.dumps(dict(
        seeds=list(seeds), rebaseline=rebaseline, split=C.SPLIT, grids=C.GRIDS,
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        sklearn=sklearn.__version__, scipy=scipy.__version__, cpu=platform.processor() or platform.machine(),
    ), indent=2, default=str))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--rebaseline", action="store_true")
    a = ap.parse_args()
    # SHAP and timing are skipped for the ablation run; they belong to the main run only
    run(a.seeds, a.out, rebaseline=a.rebaseline, do_shap=not a.rebaseline, do_timing=not a.rebaseline)
