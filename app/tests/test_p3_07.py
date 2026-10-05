import math

import torch
import torch.nn as nn

from training.config import TrainingConfig
from training.data import ManifestDataset, create_dataloader
from training.model import build_model
from training.trainer import set_reproducibility, train_epoch


def test_val_test_identical_tensors(mock_dataset_env):
    manifest_path, base_dir = mock_dataset_env
    loader_val = create_dataloader(
        manifest_path, base_dir, "val", batch_size=1, image_size=128, seed_train=43, num_workers=0
    )
    loader_test = create_dataloader(
        manifest_path, base_dir, "test", batch_size=1, image_size=128, seed_train=43, num_workers=0
    )

    tensor_val, _, _ = next(iter(loader_val))
    tensor_test, _, _ = next(iter(loader_test))
    assert torch.equal(tensor_val, tensor_test)


def test_augmentation_only_in_train(mock_dataset_env):
    manifest_path, base_dir = mock_dataset_env

    val_dataset = ManifestDataset(manifest_path, base_dir, "val", image_size=128)
    tensor_val1, _, _ = val_dataset[0]
    tensor_val2, _, _ = val_dataset[0]
    assert torch.equal(tensor_val1, tensor_val2)

    train_dataset = ManifestDataset(manifest_path, base_dir, "train", image_size=128)
    tensor_train1, _, _ = train_dataset[0]
    tensor_train2, _, _ = train_dataset[0]
    assert not torch.equal(tensor_train1, tensor_train2)


def test_dataloader_crop_id_order_same_seed(mock_dataset_env):
    manifest_path, base_dir = mock_dataset_env
    loader1 = create_dataloader(
        manifest_path,
        base_dir,
        "train",
        batch_size=1,
        image_size=128,
        seed_train=43,
        seed_aug=44,
        num_workers=0,
    )
    loader2 = create_dataloader(
        manifest_path,
        base_dir,
        "train",
        batch_size=1,
        image_size=128,
        seed_train=43,
        seed_aug=44,
        num_workers=0,
    )

    order1 = [crop_id.item() for _, _, crop_id in loader1]
    order2 = [crop_id.item() for _, _, crop_id in loader2]
    assert order1 == order2


def test_dataloader_steps_per_epoch(mock_dataset_env):
    manifest_path, base_dir = mock_dataset_env
    loader = create_dataloader(
        manifest_path,
        base_dir,
        "train",
        batch_size=2,
        image_size=128,
        seed_train=43,
        seed_aug=44,
        num_workers=0,
    )
    assert len(loader) == math.ceil(3 / 2)


def test_train_reproducible_with_same_seed(mock_dataset_env):
    manifest_path, base_dir = mock_dataset_env
    device = torch.device("cpu")

    config = TrainingConfig(
        optimizer="adam",
        batch_size=16,
        max_epochs=15,
        learning_rate=0.001,
        image_size=128,
        hidden_layers=0,
        dropout=0.0,
        seed_split=42,
        seed_train=43,
        seed_aug=44,
        seed_model=45,
        patience=5,
        min_delta=0.01,
    )

    def run_simulated_epoch():
        set_reproducibility(
            seed_model=config.seed_model, seed_train=config.seed_train, seed_split=config.seed_split
        )
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        dataloader = create_dataloader(
            manifest_path,
            base_dir,
            "train",
            batch_size=2,
            image_size=config.image_size,
            seed_train=config.seed_train,
            seed_aug=config.seed_aug,
            num_workers=0,
        )

        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        return loss, model.fc[1].weight.clone()

    loss1, weights1 = run_simulated_epoch()
    loss2, weights2 = run_simulated_epoch()

    assert loss1 == loss2
    assert torch.equal(weights1, weights2)


def test_train_epoch_updates_weights(mock_dataset_env):
    manifest_path, base_dir = mock_dataset_env
    device = torch.device("cpu")

    config = TrainingConfig(
        optimizer="adam",
        batch_size=16,
        max_epochs=15,
        learning_rate=0.001,
        image_size=128,
        hidden_layers=0,
        dropout=0.0,
        seed_split=42,
        seed_train=43,
        seed_aug=44,
        seed_model=45,
        patience=5,
        min_delta=0.01,
    )

    set_reproducibility(
        seed_model=config.seed_model, seed_train=config.seed_train, seed_split=config.seed_split
    )
    model = build_model(config).to(device)
    initial_weights = model.fc[1].weight.clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    dataloader = create_dataloader(
        manifest_path,
        base_dir,
        "train",
        batch_size=2,
        image_size=config.image_size,
        seed_train=config.seed_train,
        seed_aug=config.seed_aug,
        num_workers=0,
    )

    train_epoch(model, dataloader, optimizer, criterion, device)

    assert not torch.equal(initial_weights, model.fc[1].weight)
