# scripts/train_surrogate.py
from __future__ import annotations
import argparse
from biocircuitai.surrogate.train import TrainConfig, train_surrogate

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/processed/toggle_dataset.csv")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=str, default="128,128", help="comma-separated hidden sizes")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--out", type=str, default="data/models/surrogate.pt")
    return p.parse_args()

def main():
    args = parse_args()
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    cfg = TrainConfig(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.bs,
        lr=args.lr,
        hidden=hidden,
        dropout=args.dropout,
        model_out=args.out,
    )
    train_surrogate(cfg)

if __name__ == "__main__":
    main()
