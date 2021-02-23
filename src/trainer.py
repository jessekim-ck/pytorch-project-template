import os
import time

import torch
import torch.nn as nn

from .data import get_train_loader, get_test_loader


class Trainer(object):
    def __init__(self, args):
        self._args = args
        self._timestamp = str(time.time()).split(".")[0]

        self._LOG_DIR = "results/logs"
        self._LOG_PATH = os.path.join(self._LOG_DIR, f"{self._timestamp}.log")
        self._WEIGHT_DIR = "results/weights"
        self._WEIGHT_PATH = os.path.join(self._WEIGHT_DIR, f"{self._timestamp}.log")

        self._logging = False
        self._model = None
        self._optimizer = None
        self._train_loader = None
        self._test_loader = None

        self._init_model()

    def _init_model(self):
        """Initialize model."""
        raise NotImplementedError

    def _init_train_loader(self, csv_path, batch_size, num_workers):
        """Initialize train data loader."""
        _, self._train_loader = get_train_loader(csv_path, batch_size, num_workers)

    def _init_test_loader(self, csv_path, batch_size, num_workers):
        """Initialize test data loader."""
        _, self._test_loader = get_test_loader(csv_path, batch_size, num_workers)

    def _train_epoch(self):
        """Train model for an epoch."""
        raise NotImplementedError

    def _write_log(self, msg=""):
        """Write message to a log file and print."""
        if not self._logging:
            print(msg)
            return 0
        if not os.path.exists(self._LOG_DIR):
            os.makedirs(self._LOG_DIR)
        mode = "w" if not os.path.exists(self._LOG_PATH) else "a"
        with open(self._LOG_PATH, mode) as log:
            log.write(f"{msg}\n")
            print(msg)
        return 1

    def _save_model(self):
        """Save model state dictionary."""
        if not os.path.exists(self._WEIGHT_DIR):
            os.makedirs(self._WEIGHT_DIR)
        torch.save(self._model.module.state_dict(), self._WEIGHT_PATH)
        self._write_log(f"Saved {self._WEIGHT_PATH}.")

    def train(self):
        """Main training loop."""
        self._logging = True
        self._write_log(f"Training log for {self._timestamp}.\n")

        for key, val in self._args.__dict__.items():
            self._write_log(f"{key}: {val}")
        self._write_log()

        self._init_train_loader(self._args.csv_path, self._args.batch_size, self._args.num_workers)
        self._init_test_loader(self._args.val_csv_path, self._args.batch_size*2, self._args.num_workers)

        for epoch in range(self._args.num_epochs):
            self._write_log(f"\nEpoch {epoch + 1}")
            self._train_epoch()
            self.evaluate()
            self._save_model()

    def evaluate(self):
        """Evaluate model performance."""
        if self._test_loader is None:
            self._init_test_loader(args.csv_path, args.batch_size, args.num_workers)
        raise NotImplementedError
