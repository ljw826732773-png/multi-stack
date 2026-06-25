import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai.bc import SequenceBCPolicy
from multistack_ai.imitation import make_sequence_dataset, split_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", default=[ROOT / "results" / "expert_dataset.npz"])
    parser.add_argument("--episode-len", type=int, default=1200)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "sequence_bc_policy.pt")
    parser.add_argument("--history", type=Path, default=ROOT / "results" / "sequence_bc_training_history.csv")
    args = parser.parse_args()

    seq_parts_x, seq_parts_y = [], []
    for data_path in args.data:
        data = np.load(data_path)
        episode_lengths = data["episode_lengths"] if "episode_lengths" in data.files else None
        seq_x, seq_y = make_sequence_dataset(
            data["X"],
            data["Y"],
            episode_len=args.episode_len,
            seq_len=args.seq_len,
            stride=args.stride,
            episode_lengths=episode_lengths,
        )
        seq_parts_x.append(seq_x)
        seq_parts_y.append(seq_y)
        print(f"loaded {data_path} -> {len(seq_x)} sequence windows")
    seq_x = np.concatenate(seq_parts_x).astype(np.float32)
    seq_y = np.concatenate(seq_parts_y).astype(np.float32)
    x_train, x_val, y_train, y_val = split_dataset(seq_x, seq_y, seed=17)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = SequenceBCPolicy(hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item()) * len(xb)
        train_mse = train_loss / len(x_train)
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(torch.tensor(x_val))
            val_mse = float(loss_fn(val_pred, torch.tensor(y_val)).item())
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        print(f"epoch={epoch:03d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": args.hidden,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "sources": [str(path) for path in args.data],
        },
        args.out,
    )
    with args.history.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse"])
        writer.writeheader()
        writer.writerows(history)
    print(f"saved sequence BC policy to {args.out}")
    print(f"saved sequence BC history to {args.history}")


if __name__ == "__main__":
    main()
