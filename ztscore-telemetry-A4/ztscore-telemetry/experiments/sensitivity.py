"""Generator sensitivity check (A3 mitigation for synthetic-data validity).

Do the headline conclusions survive changes to the generator assumptions?

  1. shift magnitude k: Site B's environment is interpolated between Site A
     (k = 0, no shift) and the default Site B (k = 1), and extrapolated (k = 1.5).
  2. compromise subtlety: every kill-chain stage probability is scaled down and
     template drift is made shallower, producing quieter intrusions.

    python -m experiments.sensitivity --seeds 0 1 2 3 4 --out results/sensitivity
"""
from __future__ import annotations

import argparse, dataclasses as dc, pathlib, sys

import numpy as np
import pandas as pd

# make the project root importable when this file is run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from telemetry_gen import DEFAULT_CONFIG                                       # noqa: E402
from telemetry_gen.config import EpisodeProfile, TransportProfile, SITE_A, SITE_B  # noqa: E402
from experiments.run_experiments import run                                    # noqa: E402

MODELS = ["LogReg", "GradBoost"]   # RF omitted here for runtime; GB is the strongest learner in the main study


def lerp(a, b, k):
    """Linear interpolation: k = 0 gives a, k = 1 gives b, k > 1 extrapolates past b."""
    return a + k * (b - a)


def glerp(a, b, k):          # geometric interpolation for medians / rates
    """Interpolate on a log scale, so k = 0.5 between 10 and 40 gives 20, not 25."""
    return float(np.exp(lerp(np.log(a), np.log(b), k)))


def shifted_site_b(k: float):
    """
    A Site B profile moved fraction k of the way from Site A's environment to
    the default Site B's (k = 0: identical to A; k = 1: default B; k = 1.5: beyond B).
    """
    # transport mix: interpolate each share (keep it positive), then renormalize to sum to 1
    mix = {t: max(1e-3, lerp(SITE_A.transport_mix[t], SITE_B.transport_mix[t], k)) for t in SITE_A.transport_mix}
    z = sum(mix.values()); mix = {t: v / z for t, v in mix.items()}
    # link quality: each (median, spread) pair -> geometric for the median, linear for the spread
    tp = {t: TransportProfile(*[(glerp(a[0], b[0], k), lerp(a[1], b[1], k))
                                for a, b in zip(dc.astuple(SITE_A.transport[t]), dc.astuple(SITE_B.transport[t]))])
          for t in SITE_A.transport}
    # benign episode profiles (failover, link degradation): interpolate start rate and duration
    ep = lambda a, b: EpisodeProfile(glerp(a.p_start, b.p_start, k), lerp(a.mean_duration_h, b.mean_duration_h, k))
    # start from Site B and overwrite every environmental setting with its interpolated value
    return dc.replace(
        SITE_B, transport_mix=mix, transport=tp,
        mfa_rate=float(np.clip(lerp(SITE_A.mfa_rate, SITE_B.mfa_rate, k), 0, 1)),
        sync_age_mean_min=glerp(SITE_A.sync_age_mean_min, SITE_B.sync_age_mean_min, k),
        release_step=int(round(lerp(SITE_A.release_step, SITE_B.release_step, k))),
        upgrade_delay_mean_h=glerp(SITE_A.upgrade_delay_mean_h, SITE_B.upgrade_delay_mean_h, k),
        cellular_failover=ep(SITE_A.cellular_failover, SITE_B.cellular_failover),
        link_degradation=ep(SITE_A.link_degradation, SITE_B.link_degradation),
    )


def subtle_compromise(f: float):
    """A quieter attacker: every kill-chain stage probability is multiplied by f, and template drift moves compliance only fraction f as far from 100%."""
    cp = DEFAULT_CONFIG.compromise
    probs = {n: getattr(cp, n) * f for n in
             ("p_rollback", "p_rogue_account", "p_key_injection", "p_account_cleanup",
              "p_key_auth_sessions", "p_template_drift", "p_relocation")}
    lo, hi = cp.drift_compliance
    return dc.replace(cp, drift_compliance=(1 - (1 - lo) * f, 1 - (1 - hi) * f), **probs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/sensitivity")
    a = ap.parse_args()
    out = pathlib.Path(a.out); frames = []
    # experiment 1: rerun the full matrix with Site B moved to each shift level
    for k in (0.0, 0.5, 1.0, 1.5):
        cfg = dc.replace(DEFAULT_CONFIG, sites=(SITE_A, shifted_site_b(k)))
        m = run(a.seeds, out / f"shift_{k}", cfg=cfg, do_shap=False, do_timing=False, models=MODELS, tag=f"[shift {k}] ")
        frames.append(m.assign(experiment="shift_magnitude", level=k))
    # experiment 2: default sites, but a quieter attacker
    for f in (1.0, 0.6):
        if f == 1.0:
            continue     # identical to shift k=1.0
        cfg = dc.replace(DEFAULT_CONFIG, compromise=subtle_compromise(f))
        m = run(a.seeds, out / f"subtle_{f}", cfg=cfg, do_shap=False, do_timing=False, models=MODELS, tag=f"[subtle {f}] ")
        frames.append(m.assign(experiment="compromise_subtlety", level=f))
    # one combined table for aggregate.py
    pd.concat(frames).to_csv(out / "sensitivity_metrics.csv", index=False)
