# Safety Filter Parameter Sweep

The safety-filtered BC policy adds an interpretable correction layer around the learned behavior-cloning policy. The most important tuning parameter is `target_alpha`, which controls how quickly the filter updates its smoothed fuel-cell target.

Run:

```bash
python scripts/safety_filter_sweep.py
```

Outputs:

```text
results/safety_filter_sweep.csv
results/safety_filter_sweep.png
```

![Safety filter sweep](../results/safety_filter_sweep.png)

## Interpretation

Small `target_alpha` values make the safety layer smoother. This reduces start-stop pressure but weakens power-tracking correction. Larger values make the controller track the demand/SOC target more aggressively, which reduces power MAE but increases switching activity.

The current default value, `target_alpha=0.18`, is selected as a conservative middle point: it improves the raw BC policy's SOC margin and tracking error while keeping start-stop events below the sequential-loading baseline.
