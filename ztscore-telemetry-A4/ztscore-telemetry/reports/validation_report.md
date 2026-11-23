# Generator validation report (data/seed_0)

## 1. Label balance (row share)

| site   |   compromised |   degraded |   healthy |
|:-------|--------------:|-----------:|----------:|
| A      |        0.0355 |     0.0477 |    0.9168 |
| B      |        0.0239 |     0.1432 |    0.8329 |


Row counts by state detail:

| site   |   cellular_failover |   compromised |   healthy |   link_degradation |   maintenance |   missed_patch_window |
|:-------|--------------------:|--------------:|----------:|-------------------:|--------------:|----------------------:|
| A      |                 822 |          2386 |     61610 |               1073 |          1278 |                    31 |
| B      |                2557 |          1609 |     55969 |               2911 |          1248 |                  2906 |


## 2. Mean feature value by site and state

|                      |   ('A', 'compromised') |   ('A', 'degraded') |   ('A', 'healthy') |   ('B', 'compromised') |   ('B', 'degraded') |   ('B', 'healthy') |
|:---------------------|-----------------------:|--------------------:|-------------------:|-----------------------:|--------------------:|-------------------:|
| sw_version_delta     |                  0.076 |               0.058 |              0.035 |                  0.221 |               0.406 |              0.094 |
| version_changes_24h  |                  0.519 |               0.133 |              0.145 |                  0.585 |               0.09  |              0.144 |
| template_compliance  |                  0.946 |               0.961 |              0.995 |                  0.943 |               0.984 |              0.995 |
| policy_sync_age_min  |                  5.836 |              15.499 |              5.012 |                 16.044 |              20.209 |             14.933 |
| cert_valid           |                  1     |               1     |              1     |                  1     |               1     |              1     |
| tunnel_integrity     |                  0.993 |               0.96  |              0.995 |                  0.991 |               0.962 |              0.995 |
| latency_ms           |                 30.669 |              45.018 |             24.586 |                 71.176 |              95.819 |             61.478 |
| jitter_ms            |                  5.487 |               9.349 |              3.28  |                 16.186 |              25.868 |             12.279 |
| loss_pct             |                  0.39  |               1.45  |              0.182 |                  1.348 |               2.276 |              1.027 |
| admin_auth_strength  |                  1.618 |               1.865 |              1.859 |                  1.301 |               1.47  |              1.392 |
| account_events_24h   |                  0.925 |               1.147 |              0.516 |                  0.736 |               0.762 |              0.541 |
| ssh_key_changes_24h  |                  0.65  |               0.225 |              0.102 |                  0.477 |               0.136 |              0.099 |
| location_consistency |                  0.946 |               0.913 |              0.977 |                  0.915 |               0.898 |              0.963 |


## 3. Single-feature PR-AUC, Site A (chance = 0.036)

No single feature should approach 1.0; if one does, the task is trivial.

| feature              |   pr_auc |
|:---------------------|---------:|
| template_compliance  |    0.321 |
| version_changes_24h  |    0.223 |
| ssh_key_changes_24h  |    0.187 |
| location_consistency |    0.079 |
| admin_auth_strength  |    0.069 |
| sw_version_delta     |    0.058 |
| account_events_24h   |    0.055 |
| jitter_ms            |    0.053 |
| latency_ms           |    0.05  |
| loss_pct             |    0.048 |
| policy_sync_age_min  |    0.037 |
| tunnel_integrity     |    0.037 |
| cert_valid           |    0.036 |


## 4. Cross-site shift, healthy rows only (KS statistic, A vs B)

| feature              |   ks_stat |   p_value |
|:---------------------|----------:|----------:|
| latency_ms           |    0.6406 |    0      |
| jitter_ms            |    0.5582 |    0      |
| loss_pct             |    0.5567 |    0      |
| policy_sync_age_min  |    0.3838 |    0      |
| admin_auth_strength  |    0.3237 |    0      |
| location_consistency |    0.1057 |    0      |
| sw_version_delta     |    0.0589 |    0      |
| account_events_24h   |    0.0154 |    0      |
| tunnel_integrity     |    0.0056 |    0.3142 |
| template_compliance  |    0.0038 |    0.8004 |
| ssh_key_changes_24h  |    0.0019 |    0.9999 |
| version_changes_24h  |    0.0009 |    1      |
| cert_valid           |    0      |    1      |


Observed transport mix:

| site   |   broadband |   cellular |   mpls |
|:-------|------------:|-----------:|-------:|
| A      |       0.325 |      0.072 |  0.603 |
| B      |       0.534 |      0.311 |  0.155 |


## 5. Compromised rows by kill-chain stage

`dwell` rows carry little or no signal by design (attacker has gone quiet).

| site   |   config_manipulation |   dwell |   initial_access |   persistence |   restore |   rollback |
|:-------|----------------------:|--------:|-----------------:|--------------:|----------:|-----------:|
| A      |                   781 |    1446 |               28 |            28 |        21 |         82 |
| B      |                   482 |    1024 |               18 |            18 |        13 |         54 |

