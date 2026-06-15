# Experiment Report

This report is generated from the latest benchmark CSV files.

![Policy trade-off map](pareto_tradeoff.png)

## Initial Benchmark Leaderboard

| policy | score | h2_proxy_kg | power_mae_kw | start_stop_count | soc_min |
| --- | --- | --- | --- | --- | --- |
| Sequential | 0.6613 | 0.7131 | 1.96 | 174.88 | 0.6496 |
| Safety-Filtered DAgger | 0.5852 | 0.6301 | 11.84 | 107.50 | 0.6255 |
| Safety-Filtered BC | 0.5839 | 0.6301 | 11.84 | 111.38 | 0.6255 |
| BC Neural Policy | 0.5640 | 0.5555 | 16.05 | 54.62 | 0.5973 |
| HC-MPC-style Expert | 0.5580 | 0.6174 | 14.57 | 57.50 | 0.6194 |
| Equal | 0.5000 | 0.7204 | 1.32 | 657.00 | 0.6500 |
| DAgger Policy | 0.3226 | 0.6669 | 20.07 | 53.50 | 0.6040 |

## Cross-Cycle Average Leaderboard

| policy | score | h2_proxy_kg | power_mae_kw | start_stop_count | soc_min |
| --- | --- | --- | --- | --- | --- |
| Sequential | 0.6699 | 0.8375 | 1.11 | 15.67 | 0.6500 |
| BC Neural Policy | 0.6428 | 0.7018 | 14.18 | 16.67 | 0.6022 |
| Safety-Filtered DAgger | 0.5337 | 0.7567 | 7.34 | 42.67 | 0.6276 |
| Safety-Filtered BC | 0.5337 | 0.7567 | 7.34 | 42.67 | 0.6276 |
| HC-MPC-style Expert | 0.5198 | 0.7409 | 9.68 | 42.00 | 0.6207 |
| Equal | 0.5003 | 0.8373 | 1.10 | 44.00 | 0.6500 |
| DAgger Policy | 0.5000 | 0.6986 | 21.93 | 10.67 | 0.5740 |

## Key Observations

- Lowest hydrogen proxy: **BC Neural Policy** (0.5555).
- Best power tracking: **Equal** (1.32 kW MAE).
- Highest SOC margin: **Equal** (minimum SOC 0.6500).
- Fewest start-stop events: **DAgger Policy** (53.50).

The safety-filtered neural policy improves the raw BC policy's tracking and SOC robustness by injecting interpretable engineering constraints. The remaining trade-off is that stronger safety correction asks the fuel-cell stacks to carry more power, which can increase the hydrogen proxy relative to the raw neural policy.
