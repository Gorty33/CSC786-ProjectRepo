## Table 7 — Row recall by kill-chain stage (A-trained, source-val threshold)

|                          |   initial_access |   rollback |   restore |   persistence |   config_manipulation |   dwell |
|:-------------------------|-----------------:|-----------:|----------:|--------------:|----------------------:|--------:|
| ('A->A', 'Baseline')     |            0     |      0.527 |     0.22  |         0.807 |                 0.418 |   0.185 |
| ('A->A', 'GradBoost')    |            0     |      0.37  |     1     |         0.799 |                 0.954 |   0.199 |
| ('A->A', 'LogReg')       |            0     |      0.151 |     0     |         0.407 |                 0.796 |   0.106 |
| ('A->A', 'RandomForest') |            0     |      0.322 |     0.57  |         0.717 |                 0.938 |   0.2   |
| ('A->B', 'Baseline')     |            0.131 |      0.912 |     0.76  |         0.88  |                 0.707 |   0.299 |
| ('A->B', 'GradBoost')    |            0.04  |      0.395 |     1     |         0.858 |                 0.811 |   0.199 |
| ('A->B', 'LogReg')       |            0.04  |      0.351 |     0.137 |         0.411 |                 0.732 |   0.167 |
| ('A->B', 'RandomForest') |            0     |      0.268 |     0.69  |         0.749 |                 0.498 |   0.173 |

Share of compromised test rows by stage:

| condition   |   initial_access |   rollback |   restore |   persistence |   config_manipulation |   dwell |
|:------------|-----------------:|-----------:|----------:|--------------:|----------------------:|--------:|
| A->A        |             0.01 |      0.029 |     0.007 |          0.01 |                 0.367 |   0.575 |
| A->B        |             0.01 |      0.035 |     0.008 |          0.01 |                 0.341 |   0.594 |
