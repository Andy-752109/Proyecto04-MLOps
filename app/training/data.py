import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler

from training.preprocess import get_preprocessing_transforms


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        base_dir: str,
        split: str,
        image_size: int,
    ):
        df = pd.read_csv(manifest_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = get_preprocessing_transforms(split, image_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.base_dir, row["path"])
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)

        label = row["label"]
        return tensor, label, row["crop_id"]


def create_dataloader(
    manifest_path: str,
    base_dir: str,
    split: str,
    batch_size: int,
    image_size: int,
    seed_train: int,
    seed_aug: int | None = None,
    num_workers: int = 2,
):
    dataset = ManifestDataset(manifest_path, base_dir, split, image_size)

    g_aug = torch.Generator()
    if seed_aug is not None:
        g_aug.manual_seed(seed_aug)

        if num_workers == 0:
            torch.manual_seed(seed_aug)
    else:
        g_aug.manual_seed(0)

    if split == "train":
        g_train = torch.Generator()
        g_train.manual_seed(seed_train)

        sampler = RandomSampler(dataset, generator=g_train)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            generator=g_aug,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
