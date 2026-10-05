import copy
import os
import random

import numpy as np
import torch


def set_reproducibility(seed_model: int, seed_train: int, seed_split: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    torch.manual_seed(seed_model)
    random.seed(seed_train)
    np.random.seed(seed_split)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets, _ in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_val_macro_f1 = 0.0

    def __call__(
        self, val_loss: float, val_acc: float, val_f1: float, model: torch.nn.Module, epoch: int
    ):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_val_accuracy = val_acc
            self.best_val_macro_f1 = val_f1
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
