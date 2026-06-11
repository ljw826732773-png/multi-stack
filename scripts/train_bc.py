import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai.bc import BCPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "results" / "expert_dataset.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "bc_policy.pt")
    args = parser.parse_args()

    data = np.load(args.data)
    x_train, x_val, y_train, y_val = train_test_split(data["X"], data["Y"], test_size=0.15, random_state=7)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=256,
        shuffle=True,
    )
    model = BCPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(x_val))
            val_loss = float(loss_fn(val_pred, torch.tensor(y_val)).item())
        print(f"epoch={epoch:03d} train_mse={train_loss/len(x_train):.6f} val_mse={val_loss:.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"saved BC policy to {args.out}")


if __name__ == "__main__":
    main()

