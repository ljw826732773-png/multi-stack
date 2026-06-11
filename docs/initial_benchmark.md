# Initial Benchmark

The following table is produced by:

```bash
python scripts/generate_expert_dataset.py --episodes 8
python scripts/train_bc.py --epochs 8
python scripts/evaluate_policies.py
```

The current benchmark is intentionally lightweight. It verifies that the AI pipeline can generate expert data, train a behavior cloning policy and evaluate multiple EMS strategies in the same environment.

![Initial policy comparison](../results/policy_comparison.png)

| Policy | H2 proxy | Final SOH range | Final SOH var | SOC min | SOC max | Power MAE / kW | Start-stop count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equal | 0.7204 | 0.2000 | 0.005450 | 0.6500 | 0.6585 | 1.32 | 657.00 |
| Sequential | 0.7131 | 0.1999 | 0.005443 | 0.6496 | 0.6559 | 1.96 | 174.88 |
| HC-MPC-style Expert | 0.6174 | 0.2000 | 0.005450 | 0.6194 | 0.6498 | 14.57 | 57.50 |
| BC Neural Policy | 0.5555 | 0.2000 | 0.005450 | 0.5973 | 0.6499 | 16.05 | 54.63 |

## Interpretation

- The neural policy successfully imitates the expert allocation pattern with low supervised validation error.
- The expert-style and BC policies reduce start-stop events compared with equal allocation.
- The Python environment is designed for fast AI experimentation. The original MATLAB model remains the reference for thesis-grade numerical analysis.
- The next research step is SAC fine-tuning or constrained RL to improve tracking while preserving fuel economy and low start-stop frequency.
