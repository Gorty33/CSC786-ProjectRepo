# SD-WAN Appliance Telemetry Generator (v0.1.0)

Seeded, reproducible generator of synthetic hourly telemetry for SD-WAN edge
appliances, with healthy / degraded / compromised labels. 

No public dataset captures SD-WAN appliance telemetry with compromise labels,
which is why the data is synthetic. Results drawn from it indicate how the
scoring methods behave; they are not estimates of production performance.

## Download Requirements + Start Commands

```bash
pip install numpy pandas scipy scikit-learn matplotlib tabulate
python generate.py --seeds 0 1 2 3 4 --out data      # ~67k rows per site per seed
python validate.py --data data/seed_0 --out reports  # report + 3 figures
```

Output: `data/seed_<n>/site_A.csv`, `site_B.csv`, `manifest.json`.

## Simulation model

Each device is an hourly time series (default 168 hours, one week). Four layers
are applied in order:

1. **Baseline healthy telemetry** drawn from the site profile.
2. **Release rollout.** A new approved release is published at a site-specific
   hour; devices upgrade after a geometric delay. A device still behind the
   release after a 24-hour grace period is labeled *degraded / missed_patch_window*.
3. **Benign degraded episodes** (hard negatives): cellular failover, link
   degradation, and planned maintenance. These deliberately produce signals that
   resemble compromise (cellular transport, location drift, account events, key
   rotation, template dips).
4. **Compromise kill chain** modeled on documented UAT-8616 tradecraft. Each stage
   fires with a probability, so intrusions range from loud to quiet:

| Stage               | Observable effect                                                 | Probability            |
|---------------------|-------------------------------------------------------------------|------------------------|
| initial_access      | rogue peer account created (+1 account event)                     | 0.70                   |
| rollback            | version delta +1–3 for 2–6 h; version-change event                | 0.75                   |
| restore             | version returns; second version-change event                      | follows rollback       |
| persistence         | 1–3 SSH key changes; account deleted; admin sessions via key auth | 0.85 / 0.60 / 0.70     |
| config_manipulation | template compliance drops to 0.75–0.95 (NETCONF)                  | 0.60 (half persistent) |
| relocation          | unregistered cellular uplink, location consistency 0.1–0.5        | 0.25                   |
| dwell               | no new signal; counters decay out of the 24 h window              | remainder              |

Event-type signals are exposed as rolling 24-hour counts, so a
rollback-and-restore leaves a residuel (`version_changes_24h = 2`) after the
version itself returns to normal.

## Data dictionary

**Features** (model inputs, 14):

| Entity  | Column                                | Type  | Meaning                                               | 
|---------|---------------------------------------|-------|-------------------------------------------------------|
| Device  | `sw_version_delta`                    | int   | Releases behind the approved release                  |
| Device  | `version_changes_24h`                 | int   | Software version changes in the last 24 h             |
| Device  | `template_compliance`                 | 0–1   | Share of running config matching the applied template |
| Device  | `policy_sync_age_min`                 | float | Minutes since last OMP policy sync                    |
| Device  | `cert_valid`                          | 0/1   | Certificate / hardware identity valid                 |
| Channel | `tunnel_integrity`                    | 0–1   | Share of IPsec tunnels healthy                        |
| Channel | `transport_class`                     | cat   | Active transport: mpls / broadband / cellular         |
| Channel | `latency_ms`, `jitter_ms`, `loss_pct` | float | Raw path metrics                                      |
| User    | `admin_auth_strength`                 | 0/1/2 | Last admin session: password / key / MFA              |
| User    | `account_events_24h`                  | int   | Account create/delete events in last 24 h             |
| User    | `ssh_key_changes_24h`                 | int   | Authorized-key changes in last 24 h                   |
| User    | `location_consistency`                | 0–1   | Agreement of observed location with registration      |

**Labels:** `label` (healthy / degraded / compromised), `is_compromised` (0/1).

**Metadata (never use as features):** `site`, `device_id`, `t`,
`primary_transport`, `state_detail`, `stage`. Splits must be by `device_id`
to prevent leakage across a device's time series.

`cert_valid` is constant at 1 in v0.1.0, including for compromised devices. This
reflects the central claim of the prior paper: certificate validation does not
change when a device is compromised after onboarding.

## Cross-site shift design

The *environment* differs between sites. The *adversary*'s behavior does not. The
compromise profile is shared, so a cross-site performance drop is attributable
to environmental shift rather than a change in attacker behavior.

| Parameter                        | Site A (MPLS-heavy)       | Site B (broadband/cellular-heavy)      |
|----------------------------------|---------------------------|----------------------------------------|
| Transport mix (MPLS/BB/cell)     | 60 / 35 / 5 %             | 10 / 60 / 30 %                         |
| Link quality                     | low latency, jitter, loss | roughly 2–3× higher on every transport |
| MFA administration               | 90 % of devices           | 60 % of devices                        |
| Mean policy sync age             | 5 min                     | 15 min                                 |
| Mean upgrade delay               | 6 h                       | 24 h                                   |
| Failover / link-degradation rate | 0.4 % / 0.6 % per hour    | 1.5 % / 1.5 % per hour                 |

Channel metrics are emitted raw rather than as deviation from a per-site
baseline. This is intentional: it is the condition under which a model trained on
one site's link profile meets another's. A per-site normalization variant is a
candidate ablation. (An ablation study would show whether a normalized model 
requires absolute base-lines for cross-site generalization). 

Shift also appears in the composition of hard negatives. Site A sees almost no
missed-patch-window rows (fast patching), while Site B has thousands, so a model
trained on A has rarely seen a benign lagging device.

## Validation summary (5 seeds, 400 devices/site)

| Site | Compromised devices | Compromised row rate | Degraded row rate |
|------|---------------------|----------------------|-------------------|
| A    | 23.4 ± 3.3          | 3.2 % ± 0.4          | 5.0 % ± 0.1       |
| B    | 20.4 ± 5.6          | 2.7 % ± 0.5          | 14.2 % ± 0.8      |

Seed 0 checks (`reports/validation_report.md`):

* No single feature trivially separates compromise. The best single-feature
  PR-AUC on Site A is template compliance at 0.32 (chance 0.036).
* Healthy-traffic channel metrics shift strongly between sites (KS 0.56–0.64);
  device and management signals tied to the attack do not (KS < 0.01).
* Roughly 60 % of compromised rows are in the low-signal `dwell` stage.

## Reproducibility

* Seeds are derived per site and per device with `numpy.random.SeedSequence`,
  so changing one site's profile never alters another site's data.
* `manifest.json` records generator version, the full configuration, library
  versions, row and label counts, and a SHA-256 hash for every CSV. Regenerating
  a seed reproduces identical hashes.
* All parameters live in `telemetry_gen/config.py`.

## Known limitations

* Distributions are researcher-specified, and not fit to production telemetry.
* Telemetry is assumed honest. Adversarial falsification is out of scope.
* Compromised devices stay labeled compromised through dwell, which bounds
  achievable per-row recall. This is deliberate (the attacker is still present)
  but should be reported alongside results.
