"""Generator validation report.

Checks that the synthetic data has the properties the study depends on:
  1. base rates in the intended 1-5% range
  2. compromised and degraded states overlap (no trivially separating feature)
  3. measurable environmental shift between sites on healthy traffic
  4. the documented rollback-and-restore signature is present while
     certificate state stays constant

    python validate.py --data data/seed_0 --out reports/
"""
import argparse, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score
from telemetry_gen import FEATURES

NUMERIC = [f for f in FEATURES if f != "transport_class"]
# Direction in which each feature indicates risk (for the univariate check).
RISK_SIGN = {
    "sw_version_delta": 1, "version_changes_24h": 1, "template_compliance": -1,
    "policy_sync_age_min": 1, "cert_valid": -1, "tunnel_integrity": -1,
    "latency_ms": 1, "jitter_ms": 1, "loss_pct": 1, "admin_auth_strength": -1,
    "account_events_24h": 1, "ssh_key_changes_24h": 1, "location_consistency": -1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/seed_0")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    data, out = pathlib.Path(args.data), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(data / "site_A.csv", low_memory=False)
    B = pd.read_csv(data / "site_B.csv", low_memory=False)
    both = pd.concat([A, B])
    lines = [f"# Generator validation report ({data})\n"]

    # 1. label balance ------------------------------------------------------
    counts = both.groupby(["site", "label"]).size().unstack(fill_value=0)
    rates = (counts.T / counts.sum(axis=1)).T.round(4)
    lines += ["## 1. Label balance (row share)\n", rates.to_markdown(), "\n"]
    detail = both.groupby(["site", "state_detail"]).size().unstack(fill_value=0)
    lines += ["Row counts by state detail:\n", detail.to_markdown(), "\n"]

    # 2. per-state feature profile ---------------------------------------------
    prof = both.groupby(["site", "label"])[NUMERIC].mean().T.round(3)
    lines += ["## 2. Mean feature value by site and state\n", prof.to_markdown(), "\n"]

    # 3. univariate separability (Site A) --------------------------------------
    rows = []
    base = A.is_compromised.mean()
    for f in NUMERIC:
        score = RISK_SIGN[f] * A[f].to_numpy(dtype=float)
        rows.append((f, average_precision_score(A.is_compromised, score)))
    uni = pd.DataFrame(rows, columns=["feature", "pr_auc"]).sort_values("pr_auc", ascending=False)
    lines += [f"## 3. Single-feature PR-AUC, Site A (chance = {base:.3f})\n",
              "No single feature should approach 1.0; if one does, the task is trivial.\n",
              uni.round(3).to_markdown(index=False), "\n"]

    # 4. cross-site shift on healthy rows --------------------------------------
    hA, hB = A[A.label == "healthy"], B[B.label == "healthy"]
    ks = []
    for f in NUMERIC:
        if hA[f].nunique() > 1 or hB[f].nunique() > 1:
            res = ks_2samp(hA[f], hB[f])
            ks.append((f, res.statistic, res.pvalue))
        else:
            ks.append((f, 0.0, 1.0))
    ks = pd.DataFrame(ks, columns=["feature", "ks_stat", "p_value"]).sort_values("ks_stat", ascending=False)
    mix = both.groupby("site").transport_class.value_counts(normalize=True).unstack().round(3)
    lines += ["## 4. Cross-site shift, healthy rows only (KS statistic, A vs B)\n",
              ks.round(4).to_markdown(index=False), "\n",
              "Observed transport mix:\n", mix.to_markdown(), "\n"]

    # 5. stage coverage ----------------------------------------------------------
    st = both[both.is_compromised == 1].groupby(["site", "stage"]).size().unstack(fill_value=0)
    lines += ["## 5. Compromised rows by kill-chain stage\n",
              "`dwell` rows carry little or no signal by design (attacker has gone quiet).\n",
              st.to_markdown(), "\n"]
    (out / "validation_report.md").write_text("\n".join(lines))

    # ---- Figure 1: rollback-and-restore timeline -------------------------------
    rb = A[A.stage == "rollback"].device_id.unique()
    dev = A[A.device_id == rb[0]] if len(rb) else A[A.is_compromised == 1].iloc[[0]]
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    panels = [("sw_version_delta", "Version delta"),
              ("version_changes_24h", "Version changes (24h)"),
              ("ssh_key_changes_24h", "SSH key changes (24h)"),
              ("template_compliance", "Template compliance"),
              ("cert_valid", "Certificate valid")]
    onset = dev.loc[dev.is_compromised == 1, "t"].min()
    for ax, (col, name) in zip(axes, panels):
        ax.step(dev.t, dev[col], where="post", color="#1f4e79")
        ax.axvspan(onset, dev.t.max(), color="#c0392b", alpha=0.08)
        ax.axvline(onset, color="#c0392b", lw=1, ls="--")
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_ylim(-0.1, 1.2)
    axes[-1].set_xlabel("Hour")
    axes[0].set_title(f"Device {dev.device_id.iloc[0]}: compromise onset at hour {onset} "
                      "(shaded). Certificate never changes.", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig1_rollback_timeline.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: channel shift between sites ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, f in zip(axes, ["latency_ms", "jitter_ms", "loss_pct"]):
        bins = np.logspace(np.log10(max(both[f].min(), 1e-3)), np.log10(both[f].max()), 60)
        ax.hist(hA[f], bins=bins, alpha=0.6, weights=np.full(len(hA), 1 / len(hA)), label="Site A", color="#1f4e79")
        ax.hist(hB[f], bins=bins, alpha=0.6, weights=np.full(len(hB), 1 / len(hB)), label="Site B", color="#e67e22")
        ax.set_xscale("log"); ax.set_title(f, fontsize=10); ax.set_ylabel("Share of rows")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Healthy-traffic channel telemetry: Site A vs Site B", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_site_shift.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: hard negatives vs compromise --------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = {"healthy": "#95a5a6", "degraded": "#e67e22", "compromised": "#c0392b"}
    samp = pd.concat([g.sample(min(len(g), 1500), random_state=0) for _, g in A.groupby("label")])
    for lab in ["healthy", "degraded", "compromised"]:
        g = samp[samp.label == lab]
        jx = np.random.default_rng(0).uniform(-0.15, 0.15, len(g))
        ax.scatter(g.sw_version_delta + jx, g.template_compliance, s=6, alpha=0.5,
                   color=colors[lab], label=lab)
    ax.set_xlabel("Software version delta (jittered)")
    ax.set_ylabel("Template compliance")
    ax.set_title("Site A: degraded states overlap compromise", fontsize=10, loc="left")
    ax.legend(frameon=False, markerscale=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig3_overlap.png", dpi=200)
    plt.close(fig)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
