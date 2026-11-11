"""Synthetic SD-WAN edge-appliance telemetry with compromise labels.

Each device is simulated as an hourly time series. Layers are applied in order:

  1. baseline healthy telemetry drawn from the site profile
  2. a site-wide software release and per-device upgrade rollout
  3. benign degraded episodes (hard negatives)
  4. an optional compromise kill chain (overrides the label from onset onward)

Event-type signals (version changes, account events, SSH key changes) are
generated as per-hour events and exposed as rolling 24-hour counts, so a
rollback-and-restore leaves a residue after the version itself is restored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRANSPORTS, GeneratorConfig, SiteProfile

GENERATOR_VERSION = "0.1.0"

FEATURES = [
    # device entity
    "sw_version_delta", "version_changes_24h", "template_compliance",
    "policy_sync_age_min", "cert_valid",
    # channel entity
    "tunnel_integrity", "transport_class", "latency_ms", "jitter_ms", "loss_pct",
    # user / management entity
    "admin_auth_strength", "account_events_24h", "ssh_key_changes_24h",
    "location_consistency",
]
LABEL_COLUMNS = ["label", "is_compromised"]
META_COLUMNS = ["site", "device_id", "t", "primary_transport", "state_detail", "stage"]


def _lognormal(rng, median_sigma, size):
    median, sigma = median_sigma
    return rng.lognormal(np.log(median), sigma, size)


def _episodes(rng, n_steps, p_start, mean_duration):
    """Non-overlapping episodes as (start, end) with geometric durations."""
    out, t = [], 0
    while t < n_steps:
        if rng.random() < p_start:
            d = int(rng.geometric(1.0 / mean_duration))
            out.append((t, min(n_steps, t + d)))
            t += d
        else:
            t += 1
    return out


def _rint(rng, lo_hi):
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _rolling_sum(events, window):
    return np.convolve(events, np.ones(window, dtype=int), mode="full")[: len(events)]


def simulate_device(rng, site: SiteProfile, device_idx: int, cfg: GeneratorConfig):
    T, cp = cfg.n_steps, cfg.compromise
    t_idx = np.arange(T)

    # ---- static device attributes -------------------------------------------
    primary = rng.choice(TRANSPORTS, p=[site.transport_mix[k] for k in TRANSPORTS])
    if rng.random() < site.mfa_rate:
        auth_base = 2                                # MFA
    else:
        auth_base = int(rng.choice([0, 1], p=[0.4, 0.6]))  # password / key

    # ---- 1. baseline healthy telemetry --------------------------------------
    delta = np.zeros(T, dtype=int)
    ver_events = np.zeros(T, dtype=int)
    template = np.clip(1.0 - rng.exponential(0.005, T), 0, 1)
    sync_age = rng.exponential(site.sync_age_mean_min, T)
    cert = np.ones(T, dtype=int)
    tunnel = np.clip(1.0 - rng.exponential(0.005, T), 0, 1)
    transport = np.array([primary] * T, dtype=object)
    lat = _lognormal(rng, site.transport[primary].latency_ms, T)
    jit = _lognormal(rng, site.transport[primary].jitter_ms, T)
    loss = _lognormal(rng, site.transport[primary].loss_pct, T)
    auth = np.full(T, auth_base, dtype=int)
    acct_events = rng.poisson(0.01, T)               # routine admin activity
    key_events = rng.poisson(0.002, T)               # routine key rotation
    loc_scale = 0.08 if primary == "cellular" else 0.02
    loc = np.clip(1.0 - rng.exponential(loc_scale, T), 0, 1)

    degraded = np.zeros(T, dtype=bool)
    detail = np.array(["healthy"] * T, dtype=object)
    stage = np.array([""] * T, dtype=object)

    def set_channel(sl, tr):
        n = sl.stop - sl.start
        transport[sl] = tr
        lat[sl] = _lognormal(rng, site.transport[tr].latency_ms, n)
        jit[sl] = _lognormal(rng, site.transport[tr].jitter_ms, n)
        loss[sl] = _lognormal(rng, site.transport[tr].loss_pct, n)

    # ---- 2. release + rollout -----------------------------------------------
    r = site.release_step
    if r < T:
        upgrade_at = r + int(rng.geometric(1.0 / site.upgrade_delay_mean_h))
        delta[r:min(upgrade_at, T)] = 1
        if upgrade_at < T:
            ver_events[upgrade_at] += 1
        lagging = (delta == 1) & (t_idx - r >= cfg.patch_window_h)
        degraded |= lagging
        detail[lagging] = "missed_patch_window"

    # ---- 3. benign degraded episodes (hard negatives) -----------------------
    if primary != "cellular":
        for a, b in _episodes(rng, T, site.cellular_failover.p_start,
                              site.cellular_failover.mean_duration_h):
            sl = slice(a, b)
            set_channel(sl, "cellular")
            tunnel[sl] -= rng.uniform(0, 0.05, b - a)
            loc[sl] *= rng.uniform(0.6, 0.9, b - a)
            sync_age[sl] += rng.exponential(10, b - a)
            degraded[sl], detail[sl] = True, "cellular_failover"

    for a, b in _episodes(rng, T, site.link_degradation.p_start,
                          site.link_degradation.mean_duration_h):
        sl, n = slice(a, b), b - a
        lat[sl] *= rng.uniform(1.5, 3.0, n)
        jit[sl] *= rng.uniform(2.0, 5.0, n)
        loss[sl] += rng.uniform(1.0, 5.0, n)
        tunnel[sl] -= rng.uniform(0.02, 0.15, n)
        degraded[sl], detail[sl] = True, "link_degradation"

    for a, b in _episodes(rng, T, site.maintenance.p_start,
                          site.maintenance.mean_duration_h):
        sl, n = slice(a, b), b - a
        acct_events[a] += 1 + rng.poisson(0.5)       # legitimate admin login/changes
        if rng.random() < 0.3:
            key_events[a] += 1                        # planned key rotation
        template[sl] = np.minimum(template[sl], rng.uniform(0.85, 0.97, n))
        sync_age[sl] += rng.exponential(20, n)
        degraded[sl], detail[sl] = True, "maintenance"

    # ---- 4. compromise kill chain -------------------------------------------
    compromised = np.zeros(T, dtype=bool)
    if rng.random() < site.compromise_fraction:
        lo, hi = (int(f * T) for f in cp.onset_window)
        t0 = int(rng.integers(lo, hi))
        compromised[t0:] = True
        stage[t0:] = "dwell"
        t = t0

        # initial access: authentication bypass, optional rogue peer account
        stage[t] = "initial_access"
        if rng.random() < cp.p_rogue_account:
            acct_events[t] += 1
        t += 1

        # privilege escalation via rollback-and-restore to a vulnerable build
        if t < T and rng.random() < cp.p_rollback:
            d = _rint(rng, cp.rollback_duration_h)
            end = min(T, t + d)
            delta[t:end] += _rint(rng, cp.rollback_releases)
            ver_events[t] += 1
            stage[t:end] = "rollback"
            if end < T:
                ver_events[end] += 1                  # restore
                stage[end] = "restore"
            t = end + 1

        # persistence: injected SSH keys, account cleanup, key-based sessions
        if t < T:
            stage[t] = "persistence"
            if rng.random() < cp.p_key_injection:
                key_events[t] += _rint(rng, cp.keys_injected)
            if rng.random() < cp.p_account_cleanup:
                acct_events[t] += 1
            if rng.random() < cp.p_key_auth_sessions:
                auth[t:min(T, t + _rint(rng, cp.key_auth_duration_h))] = 1
            t += 1

        # configuration manipulation over NETCONF -> template drift
        if t < T and rng.random() < cp.p_template_drift:
            persist = rng.random() < cp.p_drift_persists
            end = T if persist else min(T, t + _rint(rng, cp.drift_duration_h))
            n = end - t
            level = rng.uniform(*cp.drift_compliance)
            template[t:end] = np.clip(level + rng.normal(0, 0.01, n), 0, 1)
            stage[t:end] = np.where(stage[t:end] == "dwell", "config_manipulation",
                                    stage[t:end])

        # optional relocation onto an unregistered uplink
        if rng.random() < cp.p_relocation:
            a = int(rng.integers(t0, T))
            b = min(T, a + _rint(rng, cp.relocation_duration_h))
            set_channel(slice(a, b), "cellular")
            loc[a:b] = rng.uniform(0.1, 0.5, b - a)

    # ---- assemble -----------------------------------------------------------
    label = np.where(compromised, "compromised",
                     np.where(degraded, "degraded", "healthy"))
    detail = np.where(compromised, "compromised", detail)

    return pd.DataFrame({
        "site": site.name,
        "device_id": f"{site.name}-{device_idx:04d}",
        "t": t_idx,
        "primary_transport": primary,
        "sw_version_delta": delta,
        "version_changes_24h": _rolling_sum(ver_events, cfg.window_h),
        "template_compliance": np.round(template, 4),
        "policy_sync_age_min": np.round(sync_age, 2),
        "cert_valid": cert,
        "tunnel_integrity": np.round(np.clip(tunnel, 0, 1), 4),
        "transport_class": transport.astype(str),
        "latency_ms": np.round(lat, 2),
        "jitter_ms": np.round(jit, 2),
        "loss_pct": np.round(np.clip(loss, 0, 100), 3),
        "admin_auth_strength": auth,
        "account_events_24h": _rolling_sum(acct_events, cfg.window_h),
        "ssh_key_changes_24h": _rolling_sum(key_events, cfg.window_h),
        "location_consistency": np.round(np.clip(loc, 0, 1), 4),
        "label": label,
        "is_compromised": compromised.astype(int),
        "state_detail": detail,
        "stage": stage,
    })


def generate_site(cfg: GeneratorConfig, site: SiteProfile, seed: int) -> pd.DataFrame:
    """Each site and each device gets an independent child seed, so changing
    one site's profile never perturbs another site's data."""
    site_idx = [s.name for s in cfg.sites].index(site.name)
    children = np.random.SeedSequence([seed, site_idx]).spawn(site.n_devices)
    frames = [simulate_device(np.random.default_rng(c), site, i, cfg)
              for i, c in enumerate(children)]
    return pd.concat(frames, ignore_index=True)


def generate(cfg: GeneratorConfig, seed: int) -> dict[str, pd.DataFrame]:
    return {s.name: generate_site(cfg, s, seed) for s in cfg.sites}
