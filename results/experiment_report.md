# Experiment Report

This report is generated from the latest benchmark CSV files.

![Policy trade-off map](pareto_tradeoff.png)

## Initial Benchmark Leaderboard

| policy | score | h2_proxy_kg | power_mae_kw | start_stop_count | soc_min |
| --- | --- | --- | --- | --- | --- |
| Sequential | 0.6492 | 0.7131 | 1.96 | 174.88 | 0.6496 |
| Safety-Filtered DAgger | 0.5713 | 0.6301 | 11.84 | 107.50 | 0.6255 |
| Safety-Filtered GRU | 0.5712 | 0.6301 | 11.84 | 107.88 | 0.6255 |
| Safety-Filtered BC | 0.5702 | 0.6301 | 11.84 | 111.38 | 0.6255 |
| BC Neural Policy | 0.5489 | 0.5555 | 16.05 | 54.62 | 0.5973 |
| HC-MPC-style Expert | 0.5429 | 0.6174 | 14.57 | 57.50 | 0.6194 |
| Equal | 0.5000 | 0.7204 | 1.32 | 657.00 | 0.6500 |
| GRU Sequence BC | 0.4545 | 0.6874 | 17.68 | 4.00 | 0.6385 |
| DAgger Policy | 0.3074 | 0.6669 | 20.07 | 53.50 | 0.6040 |

## Cross-Cycle Average Leaderboard

| policy | score | h2_proxy_kg | power_mae_kw | start_stop_count | soc_min |
| --- | --- | --- | --- | --- | --- |
| Sequential | 0.5651 | 1.3082 | 10.95 | 236.00 | 0.6498 |
| Safety-Filtered GRU | 0.5251 | 1.1873 | 35.96 | 116.40 | 0.6423 |
| Safety-Filtered DAgger | 0.5238 | 1.1873 | 35.96 | 118.80 | 0.6423 |
| Safety-Filtered BC | 0.5236 | 1.1873 | 35.96 | 119.20 | 0.6423 |
| BC Neural Policy | 0.5179 | 0.8990 | 50.22 | 58.60 | 0.5566 |
| Equal | 0.5007 | 1.3071 | 9.04 | 384.00 | 0.6500 |
| GRU Sequence BC | 0.4865 | 1.2060 | 49.02 | 4.00 | 0.6423 |
| HC-MPC-style Expert | 0.4845 | 1.1390 | 47.22 | 79.40 | 0.6316 |
| DAgger Policy | 0.4787 | 0.8096 | 54.39 | 44.40 | 0.5031 |

## Key Observations

- Lowest hydrogen proxy: **BC Neural Policy** (0.5555).
- Best power tracking: **Equal** (1.32 kW MAE).
- Highest SOC margin: **Equal** (minimum SOC 0.6500).
- Fewest start-stop events: **GRU Sequence BC** (4.00).

The raw neural policies expose a useful learning-control trade-off: they can reduce hydrogen proxy or start-stop events, but they may leave too much compensation to the battery on difficult cycles. The mixed random/EPA-trained GRU sequence policy is especially smooth, reaching the lowest start-stop count while keeping much better cross-cycle SOC margin than the earlier random-only recurrent model.

The safety-filtered variants improve deployment robustness by injecting interpretable SOC and stack-limit constraints around the learned controllers. This is why the safety-filtered BC, DAgger and GRU policies cluster together: the neural policy supplies the nominal dispatch style, while the safety layer enforces engineering feasibility.
