# Experiment Design

## Problem Formulation

The multi-stack fuel-cell EMS problem is formulated as a continuous control task. At each step, the policy observes vehicle demand, battery SOC, stack SOH values and previous stack power. It outputs four normalized stack power commands.

## Observation

```text
[P_dem, SOC, SOH_1, SOH_2, SOH_3, SOH_4, P_1_prev, P_2_prev, P_3_prev, P_4_prev, t]
```

## Action

```text
[P_1, P_2, P_3, P_4] / P_nom
```

## Reward Terms

The environment penalizes:

- hydrogen consumption proxy
- SOC deviation from the reference value
- stack SOH variance
- start-stop events
- stack power ramping
- power tracking error

## Why Behavior Cloning First

Pure reinforcement learning can be unstable in constrained energy management problems. The project therefore uses an HC-MPC-style expert to generate demonstrations and trains a neural policy through behavior cloning. This provides a stable initialization and a clear comparison point before SAC fine-tuning.

## Evaluation Metrics

- H2 consumption proxy
- final SOH range
- final SOH variance
- mean SOH
- SOC minimum and maximum
- power tracking MAE
- start-stop count
