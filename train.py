import argparse

from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--val_csv_path", type=str)
    parser.add_argument("--pretrained_weight", type=str)
    parser.add_argument("--memo", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
