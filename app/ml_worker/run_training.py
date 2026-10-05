"""Entrena un `TrainingJob` real: arma modelo/dataloaders desde su config y
manifiesto, y llama a `training.tracking.train_with_mlflow` (P3-08). Corre
dentro del subproceso que lanza `ml_worker.__main__` -- no conoce
`training_jobs` ni MariaDB, solo entrena y opcionalmente avisa por época.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

from ml_api.contracts import TrainingJob
from training.config import TrainingConfig
from training.data import create_dataloader
from training.model import build_model
from training.tracking import train_with_mlflow
from training.trainer import set_reproducibility


def dataset_paths(derived_dir: Path, dataset_release: str) -> tuple[Path, Path]:
    """`manifest.csv` y la raíz de recortes de un release, dentro de `derived_dir`.

    `derived_dir` ya es el volumen montado (`data/derived`); las rutas que trae
    `manifest_meta.json` son relativas a la raíz del repo completo, que el
    contenedor no tiene -- se reconstruyen aquí igual que en
    `ml_api.training_jobs` (mismo layout: `<derived_dir>/manifests/<release>/
    manifest.csv` y `<derived_dir>/crops`).
    """
    return derived_dir / "manifests" / dataset_release / "manifest.csv", derived_dir / "crops"


def manifest_meta_path(reports_dir: Path, dataset_release: str) -> Path:
    """`reports/manifests/<release>/manifest_meta.json`, lo que `train_with_mlflow`
    (P3-08) lee vía `MANIFEST_META_PATH` para los tags de procedencia."""
    return reports_dir / "manifests" / dataset_release / "manifest_meta.json"


def _build_optimizer(config: TrainingConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Sección 7 de docs/decisiones-proyecto3.md: "SGD usa momentum = 0.9".
    # Sin esto, MLflow registra momentum=0 (el default de torch) -- encontrado
    # real en la corrida r05 (revisión de Uriel sobre este mismo PR).
    return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)


def run_training(
    job: TrainingJob,
    *,
    manifest_path: Path,
    crops_root: Path,
    meta_path: Path,
    on_epoch=None,
    on_run_started=None,
) -> str:
    """Corre el entrenamiento completo de `job`; devuelve el `mlflow_run_id`.

    `MANIFEST_META_PATH` se define aquí, no la recibe `train_with_mlflow`
    como parámetro -- lee esa variable de entorno ella misma (P3-08). Como
    esto corre en su propio subproceso (`run_training_subprocess`), mutar el
    entorno aquí no se filtra a otros trabajos que corran en paralelo.
    """
    os.environ["MANIFEST_META_PATH"] = str(meta_path)
    config = TrainingConfig.model_validate(job.config)
    device = torch.device("cpu")
    # Presupuesto de cómputo de la sección 7: la campaña corre en un Mac de
    # 4 CPU / 8 GB -- sin este límite, torch usa todos los cores que vea.
    torch.set_num_threads(4)

    set_reproducibility(
        seed_model=config.seed_model, seed_train=config.seed_train, seed_split=config.seed_split
    )
    model = build_model(config).to(device)
    optimizer = _build_optimizer(config, model)
    criterion = nn.CrossEntropyLoss()

    # Sección 7: num_workers = 2. Corre dentro del multiprocessing.Process que
    # ya lanza ml_worker.__main__ -- multiprocessing anidado, probado real
    # (fork en Linux, que es el SO del contenedor) para confirmar que no
    # cuelga ni tranca el reporte de progreso por época.
    train_loader = create_dataloader(
        str(manifest_path),
        str(crops_root),
        "train",
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed_train=config.seed_train,
        seed_aug=config.seed_aug,
        num_workers=2,
    )
    val_loader = create_dataloader(
        str(manifest_path),
        str(crops_root),
        "val",
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed_train=config.seed_train,
        seed_aug=config.seed_aug,
        num_workers=2,
    )

    return train_with_mlflow(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        tags={"run_kind": job.run_kind, "grid_row": job.grid_row or ""},
        on_epoch=on_epoch,
        on_run_started=on_run_started,
    )


def run_training_subprocess(
    job_payload: dict, manifest_path: str, crops_root: str, meta_path: str, queue
) -> None:
    """Punto de entrada del subproceso que lanza `ml_worker.__main__` (P3-09).

    No toca `training_jobs` ni MariaDB directo: todo lo que sabe reportar
    (una época terminó, se completó, truena) va por `queue`. El proceso
    padre es el único que escribe en la base -- un solo escritor por
    trabajo evita coordinar dos conexiones concurrentes al mismo `job.id`.
    Recibe `job_payload` (no un `TrainingJob`) porque los argumentos de un
    `multiprocessing.Process` se pickle-an para cruzar al subproceso, y un
    dict corriente cruza sin depender de que Pydantic sea picklable ahí.
    """
    job = TrainingJob(**job_payload)

    def on_run_started(run_id):
        queue.put({"type": "run_started", "mlflow_run_id": run_id})

    def on_epoch(epoch, max_epochs, train_loss, val_loss, val_accuracy):
        queue.put(
            {
                "type": "progress",
                "progress": epoch / max_epochs,
                "log_line": (
                    f"época {epoch}/{max_epochs}: train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
                ),
            }
        )

    try:
        run_id = run_training(
            job,
            manifest_path=Path(manifest_path),
            crops_root=Path(crops_root),
            meta_path=Path(meta_path),
            on_epoch=on_epoch,
            on_run_started=on_run_started,
        )
    except Exception as error:  # el padre lo guarda en `training_jobs.error`, legible
        queue.put({"type": "failed", "error": str(error)})
    else:
        queue.put({"type": "completed", "mlflow_run_id": run_id})
