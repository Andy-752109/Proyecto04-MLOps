"""Recarga limpia (P3-14, paso 5): verifica que un paquete publicado se puede recargar.

Se corre en un proceso nuevo, con el `best.pt` ya descargado de S3:

1. calcula el SHA-256 del checkpoint y lo compara con el esperado (falla si difiere);
2. reconstruye el modelo con `training.model.build_model` y carga los pesos
   con `weights_only=True` (el mismo camino que `final.py` y `smoke.py`);
3. predice N recortes del split elegido con el mismo `create_dataloader`;
4. compara clase y probabilidades contra un CSV de referencia y escribe la
   tabla en Markdown para `docs/evidencia-recarga.md`.

Sale con código 1 si algo no coincide: no hay resultado "simulado".

Uso (desde `app/`, con el venv del proyecto):

    python recarga_limpia.py --version 1.0.0 \
        --checkpoint ../eval_v1.0.0/artifacts/checkpoint/best.pt \
        --expected-sha256 <sha del registry.json> \
        --config-json config_r02.json \
        --reference-csv ../reports/evaluation/predictions.csv \
        --split test --selection ../reports/selection.json

Para la 0.1.0 (smoke, no seleccionada) usa `--split val`: el test sólo lo toca
el modelo seleccionado. Si no hay un CSV de referencia, genéralo desde el
checkpoint ORIGINAL (el de MLflow) con `--write-reference`, y luego corre la
comprobación con el checkpoint descargado de S3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from safe_path import safe_path

from training.config import TrainingConfig
from training.data import create_dataloader
from training.model import build_model

CLASSES = ("cat", "dog")
PROB_TOLERANCE = 1e-6


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(checkpoint: Path, config: TrainingConfig) -> torch.nn.Module:
    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    return model


def predict_split(model, config: TrainingConfig, manifest_csv: Path, crops: Path, split: str):
    """Devuelve {crop_id: (clase_predicha, [prob_cat, prob_dog])} de todo el split."""
    loader = create_dataloader(
        str(manifest_csv),
        str(crops),
        split,
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed_train=config.seed_train,
        seed_aug=config.seed_aug,
        num_workers=0,
    )
    out: dict[str, tuple[str, list[float]]] = {}
    with torch.no_grad():
        for inputs, _targets, ids in loader:
            probs = torch.softmax(model(inputs), dim=1)
            for crop_id, row in zip(ids, probs.tolist(), strict=True):
                out[str(crop_id)] = (CLASSES[max(range(len(row)), key=row.__getitem__)], row)
    return out


def write_reference(predictions, manifest_csv: Path, split: str, path: Path) -> None:
    frame = pd.read_csv(manifest_csv)
    frame = frame[frame["split"] == split]
    sources = {str(r["crop_id"]): r["source_image_id"] for _, r in frame.iterrows()}
    truth = {str(r["crop_id"]): CLASSES[int(r["label"])] for _, r in frame.iterrows()}
    rows = [
        {
            "crop_id": cid,
            "source_image_id": sources[cid],
            "true_class": truth[cid],
            "predicted_class": cls,
            "prob_cat": probs[0],
            "prob_dog": probs[1],
        }
        for cid, (cls, probs) in predictions.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def guard_split(split: str, checkpoint_sha: str, selection_path: Path | None) -> None:
    """El test es del modelo seleccionado: otro checkpoint no debe tocarlo."""
    if split != "test":
        return
    if selection_path is None:
        sys.exit("--split test exige --selection (reports/selection.json).")
    selected = json.loads(selection_path.read_text())["checkpoint_sha256"]
    if checkpoint_sha != selected:
        sys.exit(
            "Este checkpoint no es el seleccionado (selection.json): "
            "no se evalúa sobre test. Usa --split val."
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--version", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--config-json", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument(
        "--manifest-csv", type=Path, default=Path("../data/derived/manifests/v0.1.1/manifest.csv")
    )
    p.add_argument("--crops", type=Path, default=Path("../data/derived/crops"))
    p.add_argument("--reference-csv", type=Path, required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--write-reference",
        action="store_true",
        help="genera --reference-csv desde este checkpoint y termina",
    )
    args = p.parse_args(argv)
    checkpoint, config_json = safe_path(args.checkpoint), safe_path(args.config_json)
    manifest_csv, crops = safe_path(args.manifest_csv), safe_path(args.crops)
    reference_csv = safe_path(args.reference_csv)
    selection = safe_path(args.selection) if args.selection else None

    print(f"\n### Recarga limpia - versión {args.version}\n")
    actual = sha256_of(checkpoint)
    ok_sha = actual == args.expected_sha256
    print(f"- SHA-256 esperado:  `{args.expected_sha256}`")
    print(f"- SHA-256 calculado: `{actual}` {'✅' if ok_sha else '❌'}")
    if not ok_sha:
        return 1
    guard_split(args.split, actual, selection)

    config = TrainingConfig.model_validate(json.loads(config_json.read_text()))
    model = load_model(checkpoint, config)
    predictions = predict_split(model, config, manifest_csv, crops, args.split)

    if args.write_reference:
        write_reference(predictions, manifest_csv, args.split, reference_csv)
        print(f"\nReferencia escrita: `{reference_csv}` ({len(predictions)} recortes).")
        return 0

    reference = pd.read_csv(reference_csv, dtype={"crop_id": str})
    sample = reference.sample(args.n, random_state=args.seed)
    print("\n| Recorte | Clase referencia | Clase recarga | máx \\|Δ prob\\| | ¿Coincide? |")
    print("|---|---|---|---|---|")
    all_ok = True
    for _, row in sample.iterrows():
        cid = str(row["crop_id"])
        if cid not in predictions:
            print(f"| `{cid}` | {row['predicted_class']} | (no está en el split) | - | ❌ |")
            all_ok = False
            continue
        cls, probs = predictions[cid]
        delta = max(abs(probs[0] - row["prob_cat"]), abs(probs[1] - row["prob_dog"]))
        match = cls == row["predicted_class"] and delta <= PROB_TOLERANCE
        all_ok &= match
        print(
            f"| `{cid}` | {row['predicted_class']} | {cls} | {delta:.2e} | "
            f"{'✅' if match else '❌'} |"
        )
    print(f"\nResultado: {'COINCIDE' if all_ok else 'NO COINCIDE'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
