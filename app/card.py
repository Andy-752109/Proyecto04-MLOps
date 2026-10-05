"""Genera la tarjeta de modelo (P3-14) solo a partir de datos.

Fuentes: params, tags y métricas del run (mlflow-store/mlflow.sql), reports/selection.json,
reports/manifests/<release>/counts.json y, si el run es el seleccionado, sus métricas de
test (artifacts/evaluation/ del run). Si falta un dato, falla en vez de dejar un campo vacío.

    uv run python card.py --run-id b828e032156e423097e90d797b643faa --version 0.1.0
    uv run python card.py --run-id 7e7b4a4b35464cfebb6b41714a3ad931 --version 1.0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlflow_dump import ROOT, artifacts_dir, load_run
from safe_path import safe_path, safe_run_id, safe_version

REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Falta {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def need(mapping: dict, key: str, source: str):
    value = mapping.get(key)
    if value is None or value == "":
        sys.exit(f"Falta '{key}' en {source}: la tarjeta no puede quedar con campos vacíos.")
    return value


def f4(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


COUNTS_JSON = "counts.json"
SELECTION_JSON = "selection.json"
TABLE_RULE = "|---|---|"
VAL_KEYS = (
    "best_val_accuracy",
    "best_val_macro_f1",
    "best_val_loss",
    "best_epoch",
    "stopped_epoch",
)

LOAD_MODE = [
    "## Modo de carga",
    "Desde la carpeta extraída del paquete, con `app/` del repo en el `PYTHONPATH`:",
    "",
    "```python",
    "import json",
    "import torch",
    "from training.config import TrainingConfig",
    "from training.model import build_model",
    "from training.preprocess import get_preprocessing_transforms",
    "",
    'config = TrainingConfig.model_validate(json.load(open("config.json")))',
    "model = build_model(config)",
    'model.load_state_dict(torch.load("artifacts/checkpoint/best.pt", weights_only=True))',
    "model.eval()",
    "",
    "transform = get_preprocessing_transforms("
    '"test", config.image_size)  # descrita en preprocess.json',
    'class_map = json.load(open("class_map.json"))',
    "# probs = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)",
    "# clase = class_map[str(int(probs.argmax()))]",
    "```",
    "",
]


def _facts(tags: dict, src: str) -> dict:
    """Datos del run que la tarjeta exige; falla si falta alguno."""
    facts = {key: need(tags, key, src) for key in ("release", "manifest_id")}
    facts["checkpoint_sha256"] = need(tags, "checkpoint_sha256", src)
    facts["run_kind"] = need(tags, "run_kind", src)
    classes = json.loads(need(tags, "classes", src))
    facts["classes"] = classes
    facts["class_names"] = [classes[k] for k in sorted(classes, key=int)]
    return facts


def _status(selection: dict, run_id: str, facts: dict) -> str:
    """Estado del modelo: seleccionado (y coherente con selection.json) o no."""
    selected_id = selection["run_id"]
    if selected_id != run_id:
        run_kind = facts["run_kind"]
        label = "Ensayo, no seleccionada" if run_kind == "smoke" else "No seleccionada"
        return (
            f"**{label}** (run_kind `{run_kind}`). "
            f"El modelo seleccionado es el run `{selected_id}`."
        )
    for key in ("checkpoint_sha256", "manifest_id"):
        if selection.get(key) != facts[key]:
            sys.exit(f"{SELECTION_JSON} y el run no coinciden en '{key}'.")
    candidate = need(selection, "candidate", SELECTION_JSON)
    return (
        f"**Seleccionada.** Métrica de selección "
        f"`{need(selection, 'selection_metric', SELECTION_JSON)}`, "
        f"fila `{need(candidate, 'grid_row', SELECTION_JSON)}`, "
        f"seleccionada el {need(selection, 'selected_at', SELECTION_JSON)}."
    )


def _load_counts(reports: Path, release: str) -> tuple[dict, dict]:
    counts = read_json(reports / "manifests" / release / COUNTS_JSON)
    if counts.get("release") != release:
        sys.exit(f"{COUNTS_JSON} no es del release {release}.")
    return need(counts, "totals", COUNTS_JSON), need(counts, "splits", COUNTS_JSON)


def _data_sections(
    version: str, run_id: str, tags: dict, src: str, facts: dict, totals: dict, splits: dict
) -> list[str]:
    classes, class_names = facts["classes"], facts["class_names"]
    release, manifest_id = facts["release"], facts["manifest_id"]
    out = [
        "## Identificación",
        f"- Versión del paquete: `{version}`",
        f"- Run de MLflow: `{run_id}` "
        f"(`{need(tags, 'mlflow.runName', src)}`, run_kind `{facts['run_kind']}`)",
        "",
        "## Propósito",
        f"Clasificar recortes de imagen en {len(class_names)} clases ({', '.join(class_names)}), "
        f"entrenado con el manifiesto `{manifest_id}` del release `{release}`.",
        "",
        "## Clases",
        "| Índice | Clase |",
        TABLE_RULE,
        *[f"| {k} | {classes[k]} |" for k in sorted(classes, key=int)],
        "",
        "## Datos y release",
        f"- Release del dataset: `{release}`; manifiesto `{manifest_id}` "
        f"(SHA-256 `{need(tags, 'manifest_sha256', src)}`)",
        f"- {totals['crops']} recortes de {totals['originals']} imágenes originales; por clase: "
        + ", ".join(f"{name} {info['crops']}" for name, info in totals["classes"].items()),
        "",
        "## Split",
        "| Split | Recortes | Originales | Recortes por clase |",
        "|---|---|---|---|",
    ]
    for name in ("train", "val", "test"):
        s = need(splits, name, COUNTS_JSON)
        per_class = " / ".join(f"{c}: {v['crops']}" for c, v in s["classes"].items())
        out.append(f"| {name} | {s['crops']} | {s['originals']} | {per_class} |")
    return out


def _validation_section(val: dict) -> list[str]:
    return [
        "",
        "## Métricas de validación (MLflow)",
        "| Métrica | Valor |",
        TABLE_RULE,
        *[f"| {key} | {f4(value)} |" for key, value in val.items()],
        "",
        "## Métricas de test",
    ]


def _test_section(run_id: str, class_names: list, splits: dict) -> tuple[list[str], list[str]]:
    """Métricas de test del modelo seleccionado y las limitaciones que de ellas se derivan."""
    art = artifacts_dir(run_id) / "evaluation"
    test = read_json(art / "metrics.json")
    analysis = read_json(art / "analysis.json")
    if list(test["classes"]) != class_names:
        sys.exit("Las clases de metrics.json no coinciden con las del run.")
    lines = [
        f"Split `test`, {test['total']} recortes. Accuracy **{f4(test['accuracy'])}**, "
        f"macro F1 **{f4(test['macro_f1'])}**.",
        "",
        "| Clase | Precision | Recall | F1 | Soporte |",
        "|---|---|---|---|---|",
        *[
            f"| {c} | {f4(m['precision'])} | {f4(m['recall'])} | {f4(m['f1'])} | {m['support']} |"
            for c, m in test["per_class"].items()
        ],
        "",
        "Matriz de confusión (filas: clase real, columnas: predicha):",
        "",
        "| | " + " | ".join(class_names) + " |",
        "|---|" + "---|" * len(class_names),
        *[
            f"| {class_names[i]} | " + " | ".join(str(v) for v in row) + " |"
            for i, row in enumerate(test["confusion_matrix"])
        ],
    ]
    worst = analysis["most_confused_class"]
    limitations = [
        f"El test tiene {test['total']} recortes "
        f"({splits['test']['originals']} imágenes originales): "
        f"cada error cambia la accuracy {100 / test['total']:.1f} puntos.",
        f"Clase con menor recall: `{worst}` ({f4(analysis['recall_per_class'][worst])}). "
        f"Accuracy de predecir siempre la clase mayoritaria: "
        f"{f4(analysis['baseline_majority_accuracy'])}.",
        *[
            f"Error en test: recorte `{e['crop_id']}` "
            f"({e['true_class']} -> {e['predicted_class']}, "
            f"probabilidad {e['probability']:.3f})."
            for e in analysis["errors"]
        ],
    ]
    if analysis.get("accuracy_hides_low_recall"):
        limitations.append("La accuracy oculta un recall bajo en al menos una clase.")
    return lines, limitations


def _untested_section(selected_id: str, splits: dict, val: dict) -> tuple[list[str], list[str]]:
    """Sin test: el split de test es exclusivo del modelo seleccionado."""
    lines = [
        f"No evaluada en test: el split de test es exclusivo del modelo seleccionado "
        f"(run `{selected_id}`)."
    ]
    limitations = [
        f"Sin evaluación en test: solo hay métricas de validación "
        f"({splits['val']['crops']} recortes).",
        f"Se detuvo por early stopping en la época {val['stopped_epoch']}; "
        f"los pesos son de la época {val['best_epoch']}.",
    ]
    return lines, limitations


def _environment_limitations(tags: dict, src: str) -> list[str]:
    return [
        f"Entrenado en `{need(tags, 'device', src)}` sobre `{need(tags, 'platform', src)}` "
        f"con Python "
        f"{need(tags, 'python_version', src)}, torch {need(tags, 'torch_version', src)} "
        f"y torchvision "
        f"{need(tags, 'torchvision_version', src)}; "
        f"en otro entorno puede haber diferencias numéricas mínimas.",
        "`build_model` descarga los pesos ImageNet de torchvision al construir el modelo: "
        "la carga necesita red o la caché de torch.",
    ]


def _closing_sections(
    run_id: str, tags: dict, src: str, params: dict, val: dict, ckpt_sha: str, limitations: list
) -> list[str]:
    return [
        "",
        "## Hiperparámetros",
        "| Parámetro | Valor |",
        TABLE_RULE,
        *[f"| {key} | {value} |" for key, value in sorted(params.items())],
        "",
        "## Limitaciones",
        *[f"- {item}" for item in limitations],
        "",
        "## Origen de los pesos",
        f"- Archivo: `artifacts/checkpoint/best.pt` del run `{run_id}` "
        f"(mejor época {val['best_epoch']} según validación)",
        f"- SHA-256 del checkpoint: `{ckpt_sha}`",
        f"- Código: commit `{need(tags, 'git_commit', src)}` "
        f"(git_dirty `{need(tags, 'git_dirty', src)}`)",
        "",
        *LOAD_MODE,
    ]


def build_card(run_id: str, version: str, reports: Path = REPORTS) -> str:
    run = load_run(run_id)
    params, tags, metrics = run["params"], run["tags"], run["metrics"]
    src = f"el run {run_id}"

    selection = read_json(reports / SELECTION_JSON)
    selected_id = need(selection, "run_id", SELECTION_JSON)
    selected = selected_id == run_id
    if int(version.split(".")[0]) >= 1 and not selected:
        sys.exit(
            f"La versión {version} debe ser el modelo seleccionado ({selected_id}), no {run_id}."
        )

    facts = _facts(tags, src)
    class_names, manifest_id = facts["class_names"], facts["manifest_id"]
    status = _status(selection, run_id, facts)
    totals, splits = _load_counts(reports, facts["release"])
    val = {key: need(metrics, key, src) for key in VAL_KEYS}

    out = [f"# Tarjeta de modelo - versión {version}", "", f"> {status}", ""]
    out += _data_sections(version, run_id, tags, src, facts, totals, splits)
    out += _validation_section(val)

    limitations = [
        f"Solo reconoce {len(class_names)} clases ({', '.join(class_names)}); "
        f"cualquier otra imagen "
        "se asignará a una de ellas.",
        f"Se entrenó y evaluó sobre recortes del manifiesto `{manifest_id}`, "
        f"no sobre imágenes completas; "
        "la entrada debe prepararse igual (ver `preprocess.json`).",
    ]
    if selected:
        test_lines, extra = _test_section(run_id, class_names, splits)
    else:
        test_lines, extra = _untested_section(selected_id, splits, val)
    out += test_lines
    limitations += extra + _environment_limitations(tags, src)
    out += _closing_sections(
        run_id, tags, src, params, val, facts["checkpoint_sha256"], limitations
    )

    card = "\n".join(out)
    if "None" in card:
        sys.exit("La tarjeta generada contiene 'None': revisa las fuentes de datos.")
    return card


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    args = p.parse_args(argv)

    version, run_id = safe_version(args.version), safe_run_id(args.run_id)
    reports = safe_path(args.reports_dir)
    output = safe_path(args.output or ROOT / "build" / f"model_card_{version}.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(  # NOSONAR: ruta validada con safe_path
        build_card(run_id, version, reports), encoding="utf-8"
    )
    print(f"Tarjeta v{version} escrita en {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
