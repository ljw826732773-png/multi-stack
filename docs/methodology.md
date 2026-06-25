# Methodology

This project extends a rule/MPC-based multi-stack fuel-cell energy management study with an AI-oriented pipeline.

## Strategy ladder

1. **Equal allocation** distributes fuel-cell power evenly across stacks.
2. **Sequential loading** activates stacks in a fixed order.
3. **HC-MPC-style expert** uses low-pass demand smoothing, SOC feedback and health-aware asymmetric stack allocation.
4. **Behavior cloning policy** distills the expert mapping into a neural network for fast inference.
5. **GRU sequence behavior cloning** learns history-dependent expert dispatch over rolling observation windows.
6. **DAgger imitation learning** reduces distribution shift by querying the expert on learner-visited states.
7. **Optional SAC fine-tuning** can further optimize the neural policy using a multi-objective reward.

## State

```text
[P_dem, SOC, SOH_1...SOH_4, P_1_prev...P_4_prev, normalized_time]
```

## Action

```text
[P_1, P_2, P_3, P_4] / P_nom
```

The environment clips physical power limits, stack derating limits and ramp-rate limits.

## Reward

The reward penalizes hydrogen consumption proxy, SOC deviation, SOH variance, start-stop events, power ramping and tracking error. This keeps the learning objective aligned with fuel economy, battery sustainability and stack lifetime consistency.

## Sequence Learning Extension

The GRU policy uses the same observation and action definitions, but it keeps a recurrent hidden state during rollout. Training uses rolling windows from both random expert trajectories and EPA-cycle expert trajectories, so the model learns short-horizon temporal consistency instead of treating each sample as independent. This is useful for EMS control because stack ramping, SOC drift and previous power allocation all carry information across time. Adding EPA-cycle expert data also reduces the distribution shift between random training episodes and standardized drive-cycle evaluation.
