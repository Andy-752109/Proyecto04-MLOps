"""Baseline P4-01: modelo original de P3 recargado desde el paquete, sin MLflow.

Carga `config.json` + `artifacts/checkpoint/best.pt` del paquete 1.0.0 (el que deja
`verify_reload.py` en `eval_v1.0.0/`), verifica el SHA-256 del checkpoint, predice el split
`val` del manifiesto y escribe:

- `reports/p4/quality/predictions_val_original.csv` (una fila por recorte de val);
- `reports/p4/reference/parity_reference.csv` (muestra fija de val, ambas clases, para las
  pruebas de paridad de edge y de conversión);
- un resumen JSON por stdout (y en `--summary`).

No toca MLflow ni `models/registry.json`. Se corre desde `app/`:

    uv run --group ml python -m edge_model.baseline

Sale con código 1 si falta algo, si un hash no coincide o si la accuracy recalculada no
coincide con `best_val_accuracy` de `reports/selection.json`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_CHECKPOINT_SHA256 = "f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9"
EXPECTED_MANIFEST_SHA256 = "53fc84fdaa0715f37d90324ae49be39034e2a67bc09ae0a1f1fec8425aeec80f"
CLASSES = ("cat", "dog")  # cat = 0, dog = 1 (class_map.json del paquete)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PARITY_PER_CLASS = 10
PARITY_SEED = 42

PREDICTION_COLUMNS = [
    "crop_id",
    "source_image_id",
    "true_class",
    "predicted_class",
    "prob_cat",
    "prob_dog",
]
PARITY_COLUMNS = ["crop_id", "path", "true_class", "predicted_class", "prob_cat", "prob_dog"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_original_model(config: dict) -> nn.Module:
    """ResNet-18 con la cabeza de `training.model.build_model`, sin pesos ImageNet.

    Los pesos reales vienen del checkpoint, así que no hace falta descargar IMAGENET1K_V1
    (y el baseline no depende de red ni de `training/`).
    """
    model = resnet18(weights=None)
    features = model.fc.in_features
    if config["hidden_layers"] == 1:
        model.fc = nn.Sequential(
            nn.Linear(features, 256), nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(256, 2)
        )
    else:
        model.fc = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(features, 2))
    return model


def preprocessing(image_size: int) -> transforms.Compose:
    """RGB -> Resize((S, S)) bilineal con antialias (PIL), estira, sin crop -> ToTensor ->
    normalización ImageNet. Igual a `get_preprocessing_transforms("val", S)` de P3."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_package(package_dir: Path, expected_sha: str) -> tuple[nn.Module, dict, str]:
    config_path = package_dir / "config.json"
    checkpoint = package_dir / "artifacts" / "checkpoint" / "best.pt"
    for path in (config_path, checkpoint):
        if not path.is_file():
            sys.exit(f"Falta {path}: corre verify_reload.py --version 1.0.0 primero.")
    class_map = json.loads((package_dir / "class_map.json").read_text(encoding="utf-8"))
    if [class_map[str(i)] for i in range(len(CLASSES))] != list(CLASSES):
        sys.exit(f"class_map.json del paquete no es {CLASSES}: {class_map}")
    actual = sha256_file(checkpoint)
    if actual != expected_sha:
        sys.exit(f"checkpoint_sha256 no coincide: esperado={expected_sha} archivo={actual}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = build_original_model(config)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    return model, config, actual


def read_split(manifest: Path, split: str) -> list[dict]:
    with open(manifest, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["split"] == split]


@torch.no_grad()
def predict(model: nn.Module, rows: list[dict], crops: Path, image_size: int) -> list[dict]:
    transform = preprocessing(image_size)
    out = []
    for row in rows:
        image = Image.open(crops / row["path"]).convert("RGB")
        batch = transform(image).unsqueeze(0)  # [1, 3, S, S]
        probs = torch.softmax(model(batch), dim=1)[0].tolist()
        out.append(
            {
                "crop_id": row["crop_id"],
                "source_image_id": row["source_image_id"],
                "path": row["path"],
                "true_class": CLASSES[int(row["label"])],
                "predicted_class": CLASSES[max(range(len(probs)), key=probs.__getitem__)],
                "prob_cat": probs[0],
                "prob_dog": probs[1],
            }
        )
    return out


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parity_sample(predictions: list[dict]) -> list[dict]:
    """10 recortes por clase verdadera, elegidos con semilla fija (ordenados por crop_id)."""
    rng = random.Random(PARITY_SEED)
    chosen = []
    for name in CLASSES:
        pool = [p for p in predictions if p["true_class"] == name]
        pool.sort(key=lambda p: p["crop_id"])
        chosen += rng.sample(pool, PARITY_PER_CLASS)
    return sorted(chosen, key=lambda p: p["crop_id"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--package-dir", type=Path, default=ROOT / "eval_v1.0.0")
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "derived" / "manifests" / "v0.1.1" / "manifest.csv",
    )
    p.add_argument("--crops", type=Path, default=ROOT / "data" / "derived" / "crops")
    p.add_argument("--selection", type=Path, default=ROOT / "reports" / "selection.json")
    p.add_argument("--split", default="val")
    p.add_argument(
        "--predictions-out",
        type=Path,
        default=ROOT / "reports" / "p4" / "quality" / "predictions_val_original.csv",
    )
    p.add_argument(
        "--parity-out",
        type=Path,
        default=ROOT / "reports" / "p4" / "reference" / "parity_reference.csv",
    )
    p.add_argument("--summary", type=Path, help="Además de stdout, guarda el resumen JSON aquí.")
    args = p.parse_args(argv)

    torch.set_num_threads(1)  # misma ejecución en cualquier máquina
    manifest_sha = sha256_file(args.manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        print(f"manifest.csv SHA-256 no coincide: {manifest_sha}", file=sys.stderr)
        return 1

    model, config, checkpoint_sha = load_package(args.package_dir, EXPECTED_CHECKPOINT_SHA256)
    rows = read_split(args.manifest, args.split)
    predictions = predict(model, rows, args.crops, config["image_size"])

    ids = [r["crop_id"] for r in predictions]
    correct = sum(r["true_class"] == r["predicted_class"] for r in predictions)
    accuracy = correct / len(predictions)
    expected = json.loads(args.selection.read_text(encoding="utf-8"))["candidate"][
        "best_val_accuracy"
    ]
    matches = abs(accuracy - expected) < 1e-9

    write_csv(args.predictions_out, PREDICTION_COLUMNS, predictions)
    parity = parity_sample(predictions)
    write_csv(args.parity_out, PARITY_COLUMNS, parity)

    summary = {
        "package_dir": args.package_dir.name,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": manifest_sha,
        "split": args.split,
        "rows": len(predictions),
        "unique_ids": len(set(ids)),
        "classes_present": sorted({r["true_class"] for r in predictions}),
        "correct": correct,
        "accuracy": accuracy,
        "best_val_accuracy_selection": expected,
        "accuracy_matches_selection": matches,
        "parity_rows": len(parity),
        "image_size": config["image_size"],
        "torch": torch.__version__,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")
    ok = len(set(ids)) == len(ids) == 131 and len(summary["classes_present"]) == 2 and matches
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
