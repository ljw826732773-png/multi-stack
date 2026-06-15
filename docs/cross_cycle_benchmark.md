# Cross-Cycle Benchmark

This benchmark evaluates the same EMS policies on authoritative EPA speed traces converted into a 12-ton traction-power demand profile:

- `epa_la92`: EPA LA92 Class 3 Heavy-Duty schedule, used as the main heavy-duty dynamic profile.
- `epa_us06`: EPA high-acceleration supplemental FTP schedule, used to stress transient response.
- `epa_udds`: EPA stop-and-go urban schedule, used as a city-driving reference.
- `epa_hwfet`: EPA highway fuel-economy schedule, used as a steady highway reference.
- `epa_mixed`: a concatenated EPA profile for longer multi-regime robustness checks.

The raw one-hertz text files are stored under `data/epa_cycles/`. The former synthetic `urban`, `highway` and `mixed` traces are still available for quick debugging, but they are no longer part of the default benchmark.

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
2. Does the policy reduce stack start-stop events under heavy-duty and aggressive transient loads?
3. Does the learned neural policy preserve the expert-style behavior outside the dataset distribution?

This benchmark gives the project a clearer research structure and prepares it for future SAC/DDPG or sequence-model comparisons.
