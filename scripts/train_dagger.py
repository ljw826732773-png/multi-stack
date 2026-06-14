import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai.imitation import (
    collect_dagger_queries,
    collect_expert_data,
    mean_summary,
    train_bc_model,
)


def save_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-episodes", type=int, default=4)
    parser.add_argument("--dagger-iters", type=int, default=3)
    parser.add_argument("--rollout-episodes", type=int, default=2)
    parser.add_argument("--epochs-per-iter", type=int, default=4)
    parser.add_argument("--episode-len", type=int, default=600)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "dagger_policy.pt")
    parser.add_argument("--history", type=Path, default=ROOT / "results" / "dagger_training_history.csv")
    args = parser.parse_args()

    x, y = collect_expert_data(args.init_episodes, seed=3000, episode_len=args.episode_len)
    result = train_bc_model(x, y, epochs=args.epochs_per_iter, seed=11)
    model = result.model
    history = [
        {
            "iteration": 0,
            "dataset_size": len(x),
            "train_mse": result.train_mse,
            "val_mse": result.val_mse,
            "h2_proxy_kg": "",
            "power_mae_kw": "",
            "soc_min": "",
            "start_stop_count": "",
        }
    ]
    print(f"iter=0 dataset={len(x)} train_mse={result.train_mse:.6f} val_mse={result.val_mse:.6f}")

    for iteration in range(1, args.dagger_iters + 1):
        qx, qy, summaries = collect_dagger_queries(
            model,
            episodes=args.rollout_episodes,
            seed=4000 + iteration * 100,
            episode_len=args.episode_len,
        )
        x = np.concatenate([x, qx], axis=0)
        y = np.concatenate([y, qy], axis=0)
        result = train_bc_model(x, y, epochs=args.epochs_per_iter, seed=11 + iteration, model=model)
        model = result.model
        metrics = mean_summary(summaries)
        row = {
            "iteration": iteration,
            "dataset_size": len(x),
            "train_mse": result.train_mse,
            "val_mse": result.val_mse,
            "h2_proxy_kg": metrics["h2_proxy_kg"],
            "power_mae_kw": metrics["power_mae_kw"],
            "soc_min": metrics["soc_min"],
            "start_stop_count": metrics["start_stop_count"],
        }
        history.append(row)
        print(
            "iter={iteration} dataset={dataset_size} val_mse={val_mse:.6f} "
            "h2={h2_proxy_kg:.4f} mae={power_mae_kw:.2f} soc_min={soc_min:.3f} starts={start_stop_count:.2f}".format(
                **row
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history}, args.out)
    save_history(args.history, history)
    print(f"saved DAgger policy to {args.out}")
    print(f"saved DAgger history to {args.history}")


if __name__ == "__main__":
    main()
