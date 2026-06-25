# GRU Sequence Behavior Cloning

This extension adds a recurrent imitation-learning baseline for the multi-stack EMS problem.

## Motivation

The original MLP behavior-cloning policy treats each control step as an independent state-action sample. That is useful as a compact baseline, but energy management is naturally history-dependent: ramp limits, SOC drift, previous stack power and delayed fuel-cell response all affect the next feasible command. A GRU policy gives the neural controller a hidden state so it can learn smoother temporal patterns from expert trajectories.

## Training

Run:

```bash
python scripts/generate_cycle_expert_dataset.py --repeat 2
python scripts/train_sequence_bc.py --data results/expert_dataset.npz results/epa_expert_dataset.npz --epochs 10 --seq-len 32 --stride 8
```

Generated files:

```text
results/epa_expert_dataset.npz
results/sequence_bc_policy.pt
results/sequence_bc_training_history.csv
```

The EPA expert dataset records HC-MPC-style expert trajectories on LA92, US06, UDDS, HWFET and the mixed EPA profile. The training script then combines the original random expert data with the EPA-cycle expert data and slices both sources into rolling windows. The model predicts the expert stack-power action at every step in the window, not only at the final step. This makes the recurrent model learn both local action mapping and short-horizon temporal consistency.

## Evaluation

After `results/sequence_bc_policy.pt` exists, the common evaluation scripts automatically include:

- `GRU Sequence BC`
- `Safety-Filtered GRU`

The first result is intentionally left as a raw learned controller. The second wraps the same GRU policy with the engineering safety layer. This makes the comparison useful: it shows what the sequence model learns by itself and what still needs explicit SOC and power-limit correction.

## Current Interpretation

The first GRU model trained only on random expert episodes was smooth but weak on some aggressive EPA cycles. After adding EPA-cycle expert trajectories to the training set, the raw GRU policy keeps its very low start-stop count while its cross-cycle SOC margin improves sharply. The remaining trade-off is tracking error: the recurrent policy tends to smooth stack dispatch, so the safety-filtered GRU variant remains the more deployment-oriented controller.
