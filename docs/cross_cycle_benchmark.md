# Cross-Cycle Benchmark

This benchmark evaluates the same EMS policies on three representative driving-demand profiles:

- `urban`: stop-and-go city profile with frequent acceleration and braking.
- `highway`: smoother high-speed profile with sustained load.
- `mixed`: city-to-highway combined profile.

The profiles are compact synthetic traces generated in Python. They are not intended to replace regulatory UDDS/HWFET certification files; their purpose is to provide reproducible, dependency-free tests for AI method development.

Run:

```bash
python scripts/evaluate_drive_cycles.py
```

Outputs:

```text
results/drive_cycle_benchmark.csv
results/drive_cycle_benchmark.png
```

![Cross-cycle benchmark](../results/drive_cycle_benchmark.png)

## Why This Matters

A controller that only performs well on one random profile may overfit to the demand generator. Cross-cycle testing helps separate three questions:

1. Does the policy keep SOC inside a reasonable operating band?
2. Does the policy reduce stack start-stop events under both urban and highway loads?
3. Does the learned neural policy preserve the expert-style behavior outside the dataset distribution?

This benchmark gives the project a clearer research structure and prepares it for future SAC/DDPG or sequence-model comparisons.