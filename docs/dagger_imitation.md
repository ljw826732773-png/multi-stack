# DAgger Imitation Learning Extension

The initial BC policy is trained only on expert-visited states. During closed-loop deployment, however, the learned policy may visit states that are slightly outside the expert dataset. This is the classic covariate-shift problem in behavior cloning.

This repository now includes a lightweight DAgger-style data aggregation loop:

1. Collect initial expert demonstrations from the HC-MPC-style expert.
2. Train a BC neural policy.
3. Roll out the current learner in the environment.
4. Query the expert on the states visited by the learner.
5. Aggregate the newly labeled states into the dataset and retrain.

Run:

```bash
python scripts/train_dagger.py
```

Outputs:

```text
results/dagger_policy.pt
results/dagger_training_history.csv
```

`dagger_policy.pt` is ignored by Git because model checkpoints can become large. The tracked history CSV records dataset growth and rollout behavior over iterations.

## Why This Is Deeper Than A Metric Add-on

DAgger changes the training process itself. Instead of only measuring the policy after training, it actively improves the state distribution used for learning. This makes the AI part of the project more defensible for research and internship discussion, because it addresses a real limitation of vanilla supervised imitation learning.

## Current Finding

The lightweight DAgger model is now included in the same evaluation pipeline as the rule-based, expert, BC and safety-filtered policies. The current small-scale DAgger setting reduces the validation imitation loss, but the raw DAgger policy does not yet dominate the raw BC policy in closed-loop control. This is an important research result rather than a failure: lower supervised loss is not guaranteed to produce better EMS behavior when the policy is evaluated through system dynamics.

The safety-filtered DAgger variant is therefore included as a more practical hybrid controller. It preserves the DAgger training pathway while using the safety layer to correct SOC and fuel-cell target errors during deployment. Future work should tune the DAgger rollout schedule, increase expert-query diversity and compare mixture policies where the expert and learner share control during early aggregation rounds.
