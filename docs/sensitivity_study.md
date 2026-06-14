# Sensitivity Study

This document records a lightweight parameter sweep for the HC-MPC-style expert policy. The goal is to make the project more useful as a research and resume portfolio item: the controller is not treated as a single fixed rule, but as a tunable baseline that can be compared with neural and reinforcement-learning policies.

Run the experiment with:

```bash
python scripts/sensitivity_experiment.py
```

Generated outputs:

```text
results/sensitivity_study.csv
results/sensitivity_study.png
```

![Sensitivity study](../results/sensitivity_study.png)

## Parameters

| Parameter | Meaning | Expected effect |
|---|---|---|
| `alpha` | Low-pass filter coefficient for fuel-cell power command | Larger values improve dynamic tracking but can increase ramping and start-stop pressure. |
| `soc_gain` | SOC feedback compensation gain | Larger values protect SOC more aggressively but may push fuel-cell output away from the smooth baseline. |
| `health_power` | SOH-aware allocation exponent | Larger values allocate more load to healthier stacks and protect weak stacks more strongly. |

## How To Use The Result

The sweep provides a practical bridge between the original thesis control design and the AI extension. It can be used in three ways:

1. As a stronger non-learning baseline for future SAC/DDPG comparison.
2. As an ablation study to explain which part of the expert policy matters most.
3. As a dataset-generation knob for behavior cloning, because different expert settings can produce different driving styles.

The current results should be treated as an initial engineering benchmark. The next step is to run the same sweep on real driving-cycle profiles and report mean and standard deviation across more seeds.