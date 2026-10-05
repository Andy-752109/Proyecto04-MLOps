import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    optimizer: Literal["adam", "sgd"]
    batch_size: Literal[16, 32]
    max_epochs: Literal[15, 30]
    learning_rate: float = Field(..., ge=1e-4, le=1e-2)
    image_size: Literal[128, 160]
    hidden_layers: Literal[0, 1]
    dropout: Literal[0.0, 0.3, 0.5]

    seed_split: int
    seed_train: int
    seed_aug: int
    seed_model: int

    patience: int = Field(..., ge=0)
    min_delta: float = Field(..., ge=0.0)


def export_schema(filepath: str = "training_schema.json"):
    with open(filepath, "w") as f:
        json.dump(TrainingConfig.model_json_schema(), f, indent=2)
