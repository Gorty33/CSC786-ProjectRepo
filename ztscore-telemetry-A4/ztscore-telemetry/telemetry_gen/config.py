"""Generator configuration.

Design principle: the *environment* differs between sites (transport mix,
link quality, patch cadence, admin practice), while the *adversary* does not.
The compromise profile is shared by every site, so any cross-site performance
drop measured downstream is attributable to environmental (covariate) shift,
not to a change in attacker behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TRANSPORTS = ("mpls", "broadband", "cellular")


@dataclass(frozen=True)
class TransportProfile:
    """Lognormal (median, sigma) parameters for per-hour channel metrics."""
    latency_ms: tuple[float, float]
    jitter_ms: tuple[float, float]
    loss_pct: tuple[float, float]


@dataclass(frozen=True)
class EpisodeProfile:
    """A benign degraded condition: per-hour start probability, mean duration."""
    p_start: float
    mean_duration_h: float


@dataclass(frozen=True)
class SiteProfile:
    name: str
    n_devices: int
    transport_mix: dict[str, float]          # primary transport probabilities
    transport: dict[str, TransportProfile]   # link behavior per transport
    mfa_rate: float                          # share of devices administered via MFA
    sync_age_mean_min: float                 # mean minutes since last OMP policy sync
    release_step: int                        # hour a new approved release is published
    upgrade_delay_mean_h: float              # mean hours until a device upgrades
    cellular_failover: EpisodeProfile
    link_degradation: EpisodeProfile
    maintenance: EpisodeProfile
    compromise_fraction: float               # share of devices compromised during the run


@dataclass(frozen=True)
class CompromiseProfile:
    """Kill chain modeled on documented UAT-8616 tradecraft (Cisco Talos, 2026).

    Each stage is observed with a probability, so some intrusions are loud and
    some are quiet. The same profile applies to every site.
    """
    onset_window: tuple[float, float] = (0.10, 0.85)   # onset as fraction of run
    p_rollback: float = 0.75          # rollback-and-restore to a vulnerable build
    rollback_releases: tuple[int, int] = (1, 3)        # releases rolled back
    rollback_duration_h: tuple[int, int] = (2, 6)
    p_rogue_account: float = 0.70     # account created during initial access
    p_key_injection: float = 0.85     # attacker SSH key added to admin account
    keys_injected: tuple[int, int] = (1, 3)
    p_account_cleanup: float = 0.60   # account deleted after use
    p_key_auth_sessions: float = 0.70 # admin sessions observed via injected key
    key_auth_duration_h: tuple[int, int] = (6, 72)
    p_template_drift: float = 0.60    # NETCONF configuration manipulation
    drift_compliance: tuple[float, float] = (0.75, 0.95)
    p_drift_persists: float = 0.50    # otherwise drift is reverted
    drift_duration_h: tuple[int, int] = (2, 12)
    p_relocation: float = 0.25        # device moved / unregistered uplink
    relocation_duration_h: tuple[int, int] = (6, 48)


@dataclass(frozen=True)
class GeneratorConfig:
    n_steps: int = 168                # hourly snapshots: one week
    window_h: int = 24                # rolling window for event counters
    patch_window_h: int = 24          # grace period before lagging = degraded
    sites: tuple[SiteProfile, ...] = ()
    compromise: CompromiseProfile = field(default_factory=CompromiseProfile)


# ---------------------------------------------------------------------------
# Default sites. Site A: well-provisioned, MPLS-heavy region. Site B: rural,
# broadband/cellular-heavy branches with noisier links and slower patching.
# ---------------------------------------------------------------------------
SITE_A = SiteProfile(
    name="A",
    n_devices=400,
    transport_mix={"mpls": 0.60, "broadband": 0.35, "cellular": 0.05},
    transport={
        "mpls":      TransportProfile((18, 0.25), (1.5, 0.40), (0.05, 0.60)),
        "broadband": TransportProfile((28, 0.35), (4.0, 0.50), (0.20, 0.70)),
        "cellular":  TransportProfile((55, 0.45), (12.0, 0.60), (0.80, 0.80)),
    },
    mfa_rate=0.90,
    sync_age_mean_min=5.0,
    release_step=48,
    upgrade_delay_mean_h=6.0,
    cellular_failover=EpisodeProfile(0.004, 4.0),
    link_degradation=EpisodeProfile(0.006, 3.0),
    maintenance=EpisodeProfile(0.010, 2.0),
    compromise_fraction=0.05,
)

SITE_B = SiteProfile(
    name="B",
    n_devices=400,
    transport_mix={"mpls": 0.10, "broadband": 0.60, "cellular": 0.30},
    transport={
        "mpls":      TransportProfile((35, 0.30), (3.0, 0.45), (0.10, 0.60)),
        "broadband": TransportProfile((45, 0.45), (7.0, 0.60), (0.50, 0.80)),
        "cellular":  TransportProfile((85, 0.55), (20.0, 0.70), (1.50, 0.90)),
    },
    mfa_rate=0.60,
    sync_age_mean_min=15.0,
    release_step=72,
    upgrade_delay_mean_h=24.0,
    cellular_failover=EpisodeProfile(0.015, 4.0),
    link_degradation=EpisodeProfile(0.015, 3.0),
    maintenance=EpisodeProfile(0.010, 2.0),
    compromise_fraction=0.05,
)

DEFAULT_CONFIG = GeneratorConfig(sites=(SITE_A, SITE_B))
