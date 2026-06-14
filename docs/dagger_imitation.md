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
