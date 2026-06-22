# GRU Sequence Behavior Cloning

This extension adds a recurrent imitation-learning baseline for the multi-stack EMS problem.

## Motivation

The original MLP behavior-cloning policy treats each control step as an independent state-action sample. That is useful as a compact baseline, but energy management is naturally history-dependent: ramp limits, SOC drift, previous stack power and delayed fuel-cell response all affect the next feasible command. A GRU policy gives the neural controller a hidden state so it can learn smoother temporal patterns from expert trajectories.

## Training

Run:

```bash
python scripts/train_sequence_bc.py --epochs 8 --seq-len 32 --stride 8
```

Generated files:

```text
results/sequence_bc_policy.pt
results/sequence_bc_training_history.csv
```

The script slices the expert demonstration dataset into rolling windows. The model predicts the expert stack-power action at every step in the window, not only at the final step. This makes the recurrent model learn both local action mapping and short-horizon temporal consistency.

## Evaluation

After `results/sequence_bc_policy.pt` exists, the common evaluation scripts automatically include:

- `GRU Sequence BC`
- `Safety-Filtered GRU`

The first result is intentionally left as a raw learned controller. The second wraps the same GRU policy with the engineering safety layer. This makes the comparison useful: it shows what the sequence model learns by itself and what still needs explicit SOC and power-limit correction.

## Current Interpretation

In the latest benchmark, the raw GRU policy strongly reduces stack start-stop events, but it can allow deeper SOC drawdown on aggressive EPA cycles. The safety-filtered GRU restores SOC margin and tracking behavior while preserving the recurrent policy as the base controller. This is a realistic research outcome: a sequence model captures smoother dispatch behavior, but safety constraints are still needed for deployment-grade EMS control.
