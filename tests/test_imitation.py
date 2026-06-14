from multistack_ai.imitation import collect_dagger_queries, collect_expert_data, train_bc_model


def test_collect_expert_data_and_train_small_bc_model():
    x, y = collect_expert_data(episodes=1, seed=10, episode_len=12)

    assert x.shape == (12, 11)
    assert y.shape == (12, 4)

    result = train_bc_model(x, y, epochs=1, batch_size=8, seed=3)

    assert result.train_mse >= 0.0
    assert result.val_mse >= 0.0


def test_collect_dagger_queries_uses_learner_rollout_states():
    x, y = collect_expert_data(episodes=1, seed=20, episode_len=12)
    result = train_bc_model(x, y, epochs=1, batch_size=8, seed=4)

    qx, qy, summaries = collect_dagger_queries(result.model, episodes=1, seed=30, episode_len=10)

    assert qx.shape == (10, 11)
    assert qy.shape == (10, 4)
    assert len(summaries) == 1
    assert "power_mae_kw" in summaries[0]
