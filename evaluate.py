import argparse

from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_csv_path", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--pretrained_weight", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.evaluate()
