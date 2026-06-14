# Safety-Filtered Neural Policy

The behavior cloning policy learns stack-power allocation from the HC-MPC-style expert, but a raw neural policy can still produce actions that are economically attractive while leaving too much work to the battery. To make the AI controller more practical, this project adds a lightweight safety filter around the learned policy.

The filter does not retrain the neural network. Instead, it post-processes each action using three pieces of engineering information:

1. Current vehicle demand.
2. Battery SOC deviation from the reference value.
3. Per-stack SOH derating limits.

The filtered policy keeps the neural network's allocation pattern when possible, then rescales or redistributes stack power so that the total fuel-cell command better matches the demand/SOC target.

## Why This Is Useful

Pure behavior cloning gives a fast AI baseline, but it can suffer from distribution shift and weak constraint handling. A safety layer is a practical bridge between learning-based EMS and deployable control:

- The neural model supplies a compact policy learned from expert behavior.
- The safety filter enforces interpretable physical constraints.
- The resulting policy can be compared directly against rule-based control and HC-MPC-style expert control.

This structure is also a natural stepping stone toward constrained reinforcement learning, because the same filter can be used as a fallback shield during SAC/DDPG exploration.


## Current Benchmark Observation

In the initial benchmark, the safety-filtered BC policy improves the raw BC policy's power-tracking MAE from 16.05 kW to 11.84 kW and raises the minimum SOC from 0.597 to 0.626. The trade-off is a moderate increase in hydrogen proxy because the filter asks the fuel-cell system to carry more of the vehicle demand instead of relying heavily on battery compensation.
