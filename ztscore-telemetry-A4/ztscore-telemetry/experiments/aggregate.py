"""Turn raw results into the tables and figures used in the A4 report.

    python -m experiments.aggregate --results results --out results/paper
"""
from __future__ import annotations

import argparse, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")                # draw to files only (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

# make the project root importable when this file is run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments import baseline as bl          # noqa: E402
from experiments.common import FAR_GRID         # noqa: E402

# display order, labels and colors used by every table and figure
ORDER = ["Baseline", "LogReg", "RandomForest", "GradBoost"]
LABEL = {"Baseline": "Hand-weighted", "Baseline@0.85": "Hand-weighted (θ=0.85)", "BaselineFixed": "Hand-weighted, fixed anchors", "LogReg": "Logistic reg.", "RandomForest": "Random forest", "GradBoost": "Gradient boosting"}
COLOR = {"Baseline": "#7a7974", "LogReg": "#2a78d6", "RandomForest": "#eb6834", "GradBoost": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def ms(x):
    """Format a column of per-seed values as 'mean ± SD'."""
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def gaps(m: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per seed: source gap = M(S->S) - M(S->T); transfer gap = M(T->T) - M(S->T)."""
    out = []
    # one group per (seed, method, source site): it holds the in-distribution and cross-site rows
    for (seed, model, src), g in m.groupby(["seed", "model", "source"]):
        tgt = "B" if src == "A" else "A"
        ins = g[g.test == src][metric].iloc[0]; cross = g[g.test == tgt][metric].iloc[0]
        # "native" = the same method trained at the target site and tested there (same test set as `cross`)
        native = m[(m.seed == seed) & (m.model == model) & (m.source == tgt) & (m.test == tgt)][metric].iloc[0]
        out.append(dict(seed=seed, model=model, source=src, in_dist=ins, cross=cross,
                        source_gap=ins - cross, rel_gap=(ins - cross) / ins if ins else np.nan,
                        transfer_gap=native - cross))
    return pd.DataFrame(out)


def main(res, out, main_dir="main", sens_dir="sensitivity"):
    """Read the result folders under `res` and write every table, figure and a markdown summary to `out`."""
    res, out = pathlib.Path(res), pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(res / main_dir / "metrics.csv")
    n = m.seed.nunique()                 # number of seeds in this run
    md = []                              # markdown summary, built up table by table

    # ---- table 1: detection metrics -----------------------------------------
    # one row per method x condition, each metric as mean ± SD over seeds
    t1 = []
    for model in ORDER + ["Baseline@0.85", "BaselineFixed"]:
        if model not in set(m.model):    # older result folders may lack some variants
            continue
        for cond in ["A->A", "A->B", "B->B", "B->A"]:
            g = m[(m.model == model) & (m.condition == cond)]
            t1.append({"Model": LABEL[model], "Train→Test": cond.replace("->", "→"),
                       **{k: ms(g[c]) for k, c in [("PR-AUC", "pr_auc"), ("F1", "f1"), ("Precision", "precision"),
                                                   ("Recall", "recall"), ("FAR", "far"), ("FRR", "frr"),
                                                   ("FRR (degraded)", "frr_degraded"),
                                                   ("Device recall", "device_recall")]}})
    t1 = pd.DataFrame(t1); t1.to_csv(out / "table1_detection.csv", index=False)
    md += [f"## Table 1 — Detection metrics (mean ± std over {n} seeds)\n", t1.to_markdown(index=False), ""]

    # ---- table 2: generalization gap ----------------------------------------
    rows = []
    for metric in ["pr_auc", "f1"]:
        gp = gaps(m[m.model.isin(ORDER + ["BaselineFixed"])], metric)
        gp.to_csv(out / f"gaps_{metric}.csv", index=False)
        base = gp[gp.model == "Baseline"].set_index(["seed", "source"])
        for (model, src), g in gp.groupby(["model", "source"]):
            g = g.set_index(["seed", "source"])
            # per-seed difference between this method's gap and the baseline's gap
            # (paired by seed, so both were measured on the same data)
            d = g.source_gap - base.loc[g.index].source_gap
            dt = g.transfer_gap - base.loc[g.index].transfer_gap
            learned = not model.startswith("Baseline")
            # comparison columns only make sense for learned models ("—" for baselines);
            # the Wilcoxon test needs at least 6 paired seeds
            rows.append({"Metric": metric.upper().replace("_", "-"), "Model": LABEL[model], "Source": src,
                         "In-dist": ms(g.in_dist), "Cross-site": ms(g.cross), "Source gap Δ": ms(g.source_gap),
                         "Relative gap": f"{100 * g.rel_gap.mean():.0f}%",
                         "Transfer gap": ms(g.transfer_gap),
                         "Transfer gap − baseline": ms(dt) if learned else "—",
                         "Wilcoxon p (transfer vs baseline)": f"{wilcoxon(dt).pvalue:.3f}" if learned and n >= 6 else "—",
                         "Δ − Δ_baseline": "—" if model.startswith("Baseline") else ms(d),
                         "Seeds learned gap > baseline gap": "—" if model.startswith("Baseline") else f"{int((d > 0).sum())}/{n}",
                         "Wilcoxon p (Δ vs baseline)": "—" if model.startswith("Baseline") or n < 6 else f"{wilcoxon(d).pvalue:.3f}",
                         "Seeds learned cross > baseline cross": "—" if model.startswith("Baseline") else
                             f"{int((g.cross - base.loc[g.index].cross > 0).sum())}/{n}"})
    t2 = pd.DataFrame(rows)
    # sort rows: metric, then source site, then method in display order
    t2["_o"] = t2.Model.map({LABEL[k]: i for i, k in enumerate(["BaselineFixed"] + ORDER)})
    t2 = t2.sort_values(["Metric", "Source", "_o"]).drop(columns="_o")
    t2.to_csv(out / "table2_gap.csv", index=False)
    md += ["## Table 2 — Generalization gap\n",
           "Source gap Δ = M(S→S) − M(S→T). Transfer gap = M(T→T) − M(S→T) (same test set). "
           "Positive = worse under shift.\n", t2.to_markdown(index=False), ""]

    # ---- table 3: timing ----------------------------------------------------
    tm = pd.read_csv(res / main_dir / "timing.csv")
    b = tm.set_index("model")
    # express each method's batch time as a multiple of the baseline's
    tm["Batch ×baseline"] = (tm.batch_us_per_row / b.loc["Baseline", "batch_us_per_row"]).round(2)
    tm["Model"] = tm.model.map(LABEL)
    tm = tm[["Model", "batch_us_per_row", "Batch ×baseline", "single_us"]].rename(
        columns={"batch_us_per_row": "Batch µs/decision", "single_us": "Single-call µs (incl. framework overhead)"})
    tm.to_csv(out / "table3_timing.csv", index=False)
    md += ["## Table 3 — Per-decision inference time (seed 0, Site A test, CPU)\n", tm.round(2).to_markdown(index=False), ""]

    # ---- table 4: SHAP vs expert weights ------------------------------------
    sh = pd.read_csv(res / main_dir / "shap_importance.csv")
    ew = bl.effective_weights()
    # rank agreement per seed and model: do the models rank features like the expert weights do?
    rk = []
    for (seed, model), g in sh.groupby(["seed", "model"]):
        imp = g.set_index("feature").mean_abs_shap.reindex(ew.index)     # align feature order with the weights
        rk.append(dict(seed=seed, model=model, spearman=spearmanr(imp, ew).statistic,
                       kendall=kendalltau(imp, ew).statistic))
    rk = pd.DataFrame(rk)
    # average importance over seeds, then scale each model's column to sum to 1 (comparable to weights)
    mean_imp = (sh.groupby(["model", "feature"]).mean_abs_shap.mean().unstack(0))
    norm_imp = mean_imp / mean_imp.sum()
    t4 = pd.DataFrame({"Model": [LABEL[k] for k in ORDER[1:]],
                       "Spearman ρ": [ms(rk[rk.model == k].spearman) for k in ORDER[1:]],
                       "Kendall τ": [ms(rk[rk.model == k].kendall) for k in ORDER[1:]],
                       "Top-3 SHAP features": [", ".join(norm_imp[k].sort_values(ascending=False).index[:3]) for k in ORDER[1:]]})
    t4.to_csv(out / "table4_shap_agreement.csv", index=False)
    wt = pd.concat([ew.rename("Expert weight"), norm_imp.rename(columns=LABEL)], axis=1).sort_values("Expert weight", ascending=False)
    wt.to_csv(out / "table4b_shap_vs_weights.csv")
    md += ["## Table 4 — SHAP rank agreement with expert weights (A-trained, A test)\n", t4.to_markdown(index=False), "",
           "Normalized mean |SHAP| vs expert effective weight:\n", wt.round(3).to_markdown(), ""]

    # ---- ablation & sensitivity ---------------------------------------------
    # Table 5 (only if the ablation was run): cross-site results before and after re-baselining
    abl = res / "ablation_rebaseline" / "metrics.csv"
    if abl.exists():
        a = pd.read_csv(abl)
        shared = sorted(set(a.seed) & set(m.seed))          # compare raw vs re-baselined on the same seeds
        a, mm = a[a.seed.isin(shared)], m[m.seed.isin(shared)]
        ga, gm = gaps(a[a.model.isin(ORDER)], "pr_auc"), gaps(mm[mm.model.isin(ORDER)], "pr_auc")
        md.append(f"(Table 5 uses the {len(shared)} seeds shared by both runs.)\n")
        t5 = []
        for model in ORDER:
            for src in "AB":
                x, y = gm[(gm.model == model) & (gm.source == src)], ga[(ga.model == model) & (ga.source == src)]
                t5.append({"Model": LABEL[model], "Source": src, "Cross PR-AUC (raw)": ms(x.cross),
                           "Cross PR-AUC (re-baselined)": ms(y.cross),
                           "Gap Δ (raw)": ms(x.source_gap), "Gap Δ (re-baselined)": ms(y.source_gap)})
        t5 = pd.DataFrame(t5); t5.to_csv(out / "table5_ablation.csv", index=False)
        md += ["## Table 5 — Ablation: per-site re-baselining of channel/sync metrics\n", t5.to_markdown(index=False), ""]

    # Table 6 (only if the sensitivity check was run): A-trained results at each generator setting
    sens = res / sens_dir / "sensitivity_metrics.csv"
    if sens.exists():
        s = pd.read_csv(sens)
        t6 = []
        for (exp, lvl), g in s.groupby(["experiment", "level"]):
            gp = gaps(g[g.model.isin(ORDER)], "pr_auc")
            for model, h in gp[gp.source == "A"].groupby("model"):
                # cross_se = standard error of the cross-site mean (used for the error bars)
                t6.append(dict(experiment=exp, level=lvl, model=model, in_dist=h.in_dist.mean(),
                               cross=h.cross.mean(), cross_se=h.cross.std(ddof=1) / np.sqrt(len(h)), gap=h.source_gap.mean(), gap_sd=h.source_gap.std(ddof=1),
                               transfer_gap=h.transfer_gap.mean()))
        t6 = pd.DataFrame(t6); t6.to_csv(out / "table6_sensitivity.csv", index=False)
        piv = t6.assign(Model=t6.model.map(LABEL)).pivot_table(
            index=["experiment", "level"], columns="Model", values=["in_dist", "cross", "gap"]).round(3)
        md += [f"## Table 6 — Sensitivity (A-trained, PR-AUC, mean of {s.seed.nunique()} seeds)\n", piv.to_markdown(), ""]
        fig_sensitivity(t6[t6.experiment == "shift_magnitude"], out)

    fig_gap(m, out); fig_far_frr(res / main_dir, out); fig_shap(wt, out)
    (out / "results_summary.md").write_text("\n".join(md))
    print("\n".join(md))


# ---- figures ----------------------------------------------------------------
def fig_gap(m, out):
    """Fig. 1: in-distribution vs cross-site PR-AUC bars, one panel per source site."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, src in zip(axes, "AB"):
        tgt = "B" if src == "A" else "A"
        x = np.arange(len(ORDER)); w = 0.36
        # two bars per method: solid = tested on the source site, hatched = tested on the other site
        for j, (test, hatch, lab) in enumerate([(src, None, "in-distribution"), (tgt, "////", "cross-site")]):
            g = m[(m.source == src) & (m.test == test)]
            mu = [g[g.model == k].pr_auc.mean() for k in ORDER]; sd = [g[g.model == k].pr_auc.std(ddof=1) for k in ORDER]
            ax.bar(x + (j - 0.5) * w, mu, w - 0.03, yerr=sd, capsize=2, error_kw=dict(lw=0.8, ecolor=INK2),
                   color=[COLOR[k] if j == 0 else "white" for k in ORDER],
                   edgecolor=[COLOR[k] for k in ORDER], hatch=hatch, linewidth=1.2, label=lab)
        # dotted line at the compromised-row base rate (what random scoring would get)
        base = m[m.test == tgt].base_rate.mean()
        ax.axhline(base, color=INK2, lw=0.8, ls=":")
        ax.text(len(ORDER) - 0.5, base + 0.01, "chance", fontsize=7, color=INK2, ha="right")
        ax.set_xticks(x, [LABEL[k].replace(" ", "\n") for k in ORDER], fontsize=7.5)
        ax.set_title(f"Trained on Site {src}: {src}→{src} vs {src}→{tgt}", loc="left", fontsize=9, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("PR-AUC, compromised class")
    # shared legend above both panels (gray swatches stand for every method's color)
    h = [plt.Rectangle((0, 0), 1, 1, fc=INK2, ec=INK2), plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK2, hatch="////")]
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(h, ["in-distribution (solid)", "cross-site (hatched)   ·   error bars = ±1 SD across seeds"],
               frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out / "fig_gap.png"); plt.close(fig)


def fig_far_frr(res, out):
    """Fig. 2: FAR vs FRR trade-off curves (averaged over seeds) for A->A and A->B."""
    cv = pd.read_json(res / "far_frr_curves.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, (src, test) in zip(axes, [("A", "A"), ("A", "B")]):
        for k in ORDER:
            g = cv[(cv.source == src) & (cv.test == test) & (cv.model == k)]
            frr = np.vstack(g.frr.values).mean(0)          # average the per-seed curves point by point
            ax.plot(FAR_GRID, frr, color=COLOR[k], lw=2 if k != "Baseline" else 1.6,
                    ls="--" if k == "Baseline" else "-", label=LABEL[k])
        # log scale on FRR so the low-false-reject region (where methods differ) is visible
        ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_ylim(1e-3, 1)
        ax.set_xlabel("False-accept rate (compromised rows allowed)")
        ax.set_title(f"{src}→{test}", loc="left", fontsize=9)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("False-reject rate\n(benign rows denied)")
    axes[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(out / "fig_far_frr.png"); plt.close(fig)


def fig_shap(wt, out):
    """Fig. D1: expert weight vs normalized SHAP importance, grouped bars per feature."""
    wt = wt.iloc[::-1]                   # reverse so the highest expert weight is drawn at the top
    cols = ["Expert weight"] + [LABEL[k] for k in ORDER[1:]]
    colors = [COLOR["Baseline"]] + [COLOR[k] for k in ORDER[1:]]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(wt)); h = 0.2
    # four bars per feature, offset vertically: expert weight, then each learned model
    for j, (c, col) in enumerate(zip(cols, colors)):
        ax.barh(y + (j - 1.5) * h, wt[c], h * 0.9, color=col, label=c)
    ax.set_yticks(y, wt.index, fontsize=7.5); ax.set_xlabel("Share of total (expert weight or normalized mean |SHAP|)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_shap.png"); plt.close(fig)


def fig_sensitivity(t, out):
    """Fig. 3: cross-site PR-AUC of A-trained methods as Site B's shift magnitude grows."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for k in ORDER:
        g = t[t.model == k].sort_values("level")
        if g.empty:                      # the sensitivity check skips random forest
            continue
        ax.errorbar(g.level, g.cross, yerr=g.cross_se, color=COLOR[k], marker="o", ms=4, lw=1.8, capsize=2,
                    ls="--" if k == "Baseline" else "-", label=LABEL[k])
    ax.set_xlabel("Shift magnitude k (0 = Site B drawn like Site A)"); ax.set_ylabel("Cross-site PR-AUC (A→B), ±1 SE")
    ax.set_xticks([0, 0.5, 1, 1.5]); ax.set_ylim(0, None); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_sensitivity.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="results/paper")
    ap.add_argument("--main", default="main"); ap.add_argument("--sens", default="sensitivity")
    a = ap.parse_args(); main(a.results, a.out, a.main, a.sens)
