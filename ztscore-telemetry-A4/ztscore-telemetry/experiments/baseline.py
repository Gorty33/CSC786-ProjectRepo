"""Hand-weighted additive trust score (the CSC773 ZT-Score-SD-WAN baseline).

Implements Eqs. 1-3 of the prior paper:

    E_k   = sum_i w_i * a_i          a_i in [0,1], sum_i w_i = 1     (entity score)
    T     = w_d E_d + w_c E_c + w_u E_u                              (composite)
    theta = theta_0 + beta * s(r)                                    (dynamic threshold)

Access is granted only if T >= theta AND every entity score clears its floor.

Every weight, floor and threshold below is set a priori from the prior paper's
design reasoning, not fit to data. Attribute normalizations use absolute
anchors (an operator-written rule set). Like the learned models, the baseline
is calibrated on its source site: the site-relative link/sync metrics are
rescaled to a reference network before scoring (label-free; see calibrate()).
Passing calibration=None reproduces the fixed, as-written anchors.

To evaluate the baseline with threshold-free metrics (PR-AUC) and threshold
sweeps, the floor-gated decision is expressed as one continuous score:

    S = min(T, E_d + (theta - f_d), E_c + (theta - f_c), E_u + (theta - f_u))

At any threshold theta, S < theta  <=>  T < theta  or  some E_k < f_k, so the
decision is exactly the paper's rule. Risk = 1 - S.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- expert weights (a priori) ----------------------------------------------
# how much each entity contributes to the composite score T (Eq. 2). Sums to 1
ENTITY_WEIGHTS = {"device": 0.45, "channel": 0.20, "user": 0.35}

# how much each attribute contributes to its entity score E_k (Eq. 1).
# Each entity's attribute weights sum to 1
ATTR_WEIGHTS = {
    "device": {   # software version + template compliance carry the heaviest weight (CSC773 Sec. V-C)
        "sw_version_delta": 0.30, "version_changes_24h": 0.25,
        "template_compliance": 0.30, "policy_sync_age_min": 0.10, "cert_valid": 0.05,
    },
    "channel": {
        "tunnel_integrity": 0.30, "transport_class": 0.20,
        "latency_ms": 0.15, "jitter_ms": 0.15, "loss_pct": 0.20,
    },
    "user": {
        "admin_auth_strength": 0.25, "account_events_24h": 0.25,
        "ssh_key_changes_24h": 0.30, "location_consistency": 0.20,
    },
}

# minimum score each entity must reach on its own, so a strong channel/user score cannot hide a failing device
FLOORS = {"device": 0.50, "channel": 0.50, "user": 0.50}   # device floor 0.50 from CSC773 worked example

# ---- site calibration (label-free) ------------------------------------------
# the absolute link anchors in normalize() were written for a reference network with these medians (all rows, clean MPLS/broadband profile). 
# Like the learned models, the baseline is calibrated on its SOURCE site's training data: each site-relative metric is rescaled by REF / source median before scoring. 
# No labels are used. calibration=None reproduces the fixed, as-written anchors
CALIBRATED_COLS = ["latency_ms", "jitter_ms", "loss_pct", "policy_sync_age_min"]
REF_MEDIANS = {"latency_ms": 21.0, "jitter_ms": 2.09, "loss_pct": 0.082, "policy_sync_age_min": 3.61}


def calibrate(df_source: pd.DataFrame) -> dict:
    """Source-site medians of the site-relative metrics (label-free)."""
    return {c: float(df_source[c].median()) for c in CALIBRATED_COLS}


def apply_calibration(df: pd.DataFrame, calibration: dict | None) -> pd.DataFrame:
    """Rescale each calibrated column so the source site's median lines up with
    the reference median the anchors were written for. None = no rescaling."""
    if calibration is None:          # fixed-anchor control: score the raw telemetry
        return df
    out = df.copy()                  # never modify the caller's DataFrame
    for c in CALIBRATED_COLS:
        # e.g. a source site whose median latency is 2x the reference has every
        # latency value halved, so "normal for this site" maps to "normal" for the rules
        out[c] = out[c] * (REF_MEDIANS[c] / calibration[c])
    return out


# dynamic threshold (Eq. 3): theta = theta_0 + beta * sensitivity.
# Sensitivity 0 (branch access) -> 0.50; sensitivity 1 (management plane) -> 0.85
THETA_0, BETA = 0.50, 0.35       # low-sensitivity branch 0.50, management plane 0.85 (CSC773 Sec. VI-B)
THETA_MGMT = THETA_0 + BETA * 1.0


def effective_weights() -> pd.Series:
    """Composite weight of each raw attribute, w_entity * w_attr (sums to 1)."""
    # flatten the two-level weight tables into one weight per attribute; this is
    # what SHAP importances are compared against
    return pd.Series({a: ENTITY_WEIGHTS[e] * w
                      for e, attrs in ATTR_WEIGHTS.items() for a, w in attrs.items()})


def _lin_bad(x, good, bad):
    """1 at or below `good`, 0 at or above `bad`, linear between."""
    return np.clip((bad - x) / (bad - good), 0.0, 1.0)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw telemetry to trust attributes in [0,1] (1 = trustworthy)."""
    a = pd.DataFrame(index=df.index)

    # device attributes
    # releases behind the approved one: 0 -> 1.0, 3 or more -> 0.0 (rollback shows up here)
    a["sw_version_delta"] = 1 - np.minimum(df.sw_version_delta, 3) / 3
    # each software change in 24 h costs 0.4 (rollback-and-restore produces several)
    a["version_changes_24h"] = np.maximum(0, 1 - 0.4 * df.version_changes_24h)
    # template compliance below 70% scores 0; 100% scores 1 (NETCONF drift lowers it)
    a["template_compliance"] = np.clip((df.template_compliance - 0.70) / 0.30, 0, 1)
    # trust decays as the last policy sync gets older (30-minute time constant)
    a["policy_sync_age_min"] = np.exp(-df.policy_sync_age_min / 30.0)
    a["cert_valid"] = df.cert_valid.astype(float)

    # channel attributes
    a["tunnel_integrity"] = np.clip((df.tunnel_integrity - 0.80) / 0.20, 0, 1)
    # transport class is scored by how controlled the path is
    a["transport_class"] = df.transport_class.map({"mpls": 1.0, "broadband": 0.8, "cellular": 0.4}).astype(float)
    # link quality: fully trusted at or below the first value, zero at or above the second
    a["latency_ms"] = _lin_bad(df.latency_ms, 30, 200)
    a["jitter_ms"] = _lin_bad(df.jitter_ms, 5, 50)
    a["loss_pct"] = _lin_bad(df.loss_pct, 0.5, 5.0)

    # user attributes
    # 0 = password only, 1 = key, 2 = MFA
    a["admin_auth_strength"] = df.admin_auth_strength.map({0: 0.2, 1: 0.6, 2: 1.0}).astype(float)
    # each account created/deleted in 24 h costs 0.35; each SSH-key change costs 0.45
    a["account_events_24h"] = np.maximum(0, 1 - 0.35 * df.account_events_24h)
    a["ssh_key_changes_24h"] = np.maximum(0, 1 - 0.45 * df.ssh_key_changes_24h)
    a["location_consistency"] = df.location_consistency.astype(float)
    return a


def entity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Eq. 1: weighted sum of normalized attributes, one column per entity."""
    a = normalize(df)
    # for each entity, sum (attribute weight x normalized attribute) over its attributes
    return pd.DataFrame({e: sum(w * a[k] for k, w in attrs.items())
                         for e, attrs in ATTR_WEIGHTS.items()}, index=df.index)


def trust_score(df: pd.DataFrame, theta: float = THETA_MGMT, calibration: dict | None = None) -> np.ndarray:
    """Floor-gated composite S (higher = more trusted)."""
    E = entity_scores(apply_calibration(df, calibration))
    T = sum(ENTITY_WEIGHTS[e] * E[e] for e in E)            # Eq. 2: composite score
    # shift each entity score so that "E_k below its floor" becomes "below theta".
    # Taking the minimum with T then denies access if EITHER the composite fails
    # OR any single entity fails its floor, exactly the paper's rule
    gated = [E[e] + (theta - FLOORS[e]) for e in E]
    return np.minimum.reduce([T.values] + [g.values for g in gated])


def risk_score(df: pd.DataFrame, theta: float = THETA_MGMT, calibration: dict | None = None) -> np.ndarray:
    """1 - S, so that higher = more likely compromised (comparable to model probabilities)."""
    return 1.0 - trust_score(df, theta, calibration)
