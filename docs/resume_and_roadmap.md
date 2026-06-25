# Resume Notes and Roadmap

## Resume Bullets

- Built a Gymnasium-based multi-stack fuel-cell hybrid energy management environment with SOC dynamics, stack SOH degradation, start-stop penalties and ramp-rate constraints.
- Designed a health-aware HC-MPC-style expert controller for asymmetric stack power allocation and lifetime-aware load shifting.
- Generated expert demonstrations and trained a PyTorch behavior cloning policy to approximate continuous stack power allocation.
- Added a GRU sequence-imitation policy trained on mixed random and EPA-cycle expert data to model history-dependent dispatch behavior under ramp-rate and SOC dynamics.
- Compared equal allocation, sequential loading, expert control and neural policy using hydrogen consumption proxy, SOC range, SOH variance, power tracking error and start-stop count.
- Prepared an optional SAC reinforcement-learning entrypoint for future policy fine-tuning under multi-objective rewards.

## Interview Story

This project starts from a conventional multi-stack fuel-cell EMS simulation and extends it into an AI control platform. The original control objective is not only to meet vehicle power demand, but also to reduce hydrogen consumption and improve lifetime consistency among stacks with different initial SOH.

The AI extension uses the HC-MPC-style controller as an expert. A neural policy first learns the expert state-action mapping through behavior cloning, which gives a stable initialization. The project now includes both a single-step MLP policy and a GRU sequence policy. The GRU path also shows a useful data story: adding EPA-cycle expert trajectories improves cross-cycle SOC generalization, while the recurrent policy still tends to smooth stack dispatch and benefits from explicit safety correction. Reinforcement learning can then be used as a second stage to fine-tune the policy for better trade-offs among hydrogen economy, SOC stability and SOH balancing.

## Near-Term Roadmap

1. **MATLAB-to-Python data bridge**
   - Export real HC-MPC trajectories from MATLAB.
   - Train BC with thesis-grade expert data instead of the lightweight Python expert.

2. **Sequence-model refinement**
   - Compare GRU, temporal CNN and small Transformer policies.
   - Compare random-only training with mixed random/EPA expert training.
   - Add ablations for sequence length and rollout horizon.

3. **SAC fine-tuning**
   - Initialize policy around the BC solution.
   - Penalize constraint violations more strongly.
   - Compare BC and SAC under the same random driving cycles.

4. **Constrained RL**
   - Add safety layers for SOC and stack power limits.
   - Evaluate whether constrained RL improves power tracking without increasing start-stop events.

5. **Multi-agent extension**
   - Treat each stack as an agent.
   - Use centralized training and decentralized execution.
   - Study whether cooperative policies improve SOH consistency.

6. **Richer result reporting**
   - Add time-series plots for SOC, stack power and SOH.
   - Add inference-time comparison between expert optimization and neural policy.
