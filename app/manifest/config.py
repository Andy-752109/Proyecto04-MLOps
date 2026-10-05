"""Configuración validada del manifiesto (`manifest.yaml`)."""

from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from splits.models import Ratio, SplitsConfig


class ManifestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    release: Annotated[str, Field(pattern=r"^v\d+\.\d+\.\d+$")]
    train: Ratio
    val: Ratio
    test: Ratio
    seed: int
    classes: Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=2)]
    max_deviation_pp: Annotated[float, Field(gt=0, le=100)]

    @model_validator(mode="after")
    def classes_are_unique(self) -> Self:
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(f"classes tiene nombres repetidos: {self.classes}")
        return self

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> Self:
        # Misma regla que el split de P2; se reutiliza para no divergir.
        self.to_splits_config()
        return self

    def to_splits_config(self) -> SplitsConfig:
        return SplitsConfig(train=self.train, val=self.val, test=self.test, seed=self.seed)

    @property
    def ratios(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}

    @property
    def labels(self) -> dict[str, int]:
        """Nombre de clase → índice del modelo, en el orden de `classes`."""
        return {name: index for index, name in enumerate(self.classes)}


def load_manifest_config(path: Path | None = None) -> ManifestConfig:
    config_path = path if path is not None else Path(__file__).with_name("manifest.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ManifestConfig.model_validate(raw)
