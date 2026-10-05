import subprocess

import mlflow
import pandas as pd
import pytest
import torch
import torch.nn as nn
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from training.config import TrainingConfig
from training.data import create_dataloader
from training.model import build_model
from training.tracking import compute_sha256, get_git_info, train_with_mlflow
from training.trainer import EarlyStopping


def test_seed_aug_workers_effect(mock_dataset_env):
    manifest_path, base_dir = mock_dataset_env

    def get_batch(seed_aug_val, nw_val):
        loader = create_dataloader(
            manifest_path,
            base_dir,
            "train",
            batch_size=2,
            image_size=128,
            seed_train=43,
            seed_aug=seed_aug_val,
            num_workers=nw_val,
        )
        return next(iter(loader))

    for nw in [0, 2]:
        t1, _, crop1 = get_batch(44, nw)
        t2, _, crop2 = get_batch(44, nw)

        assert torch.equal(t1, t2), (
            f"Fallo con num_workers={nw}: Tensores distintos con misma semilla"
        )
        assert torch.equal(crop1, crop2)

        t3, _, crop3 = get_batch(99, nw)

        assert not torch.equal(t1, t3), (
            f"Fallo con num_workers={nw}: Tensores idénticos con distinta semilla"
        )
        assert torch.equal(crop1, crop3), f"Fallo con num_workers={nw}: El orden cambió"


def test_early_stopping_controlled_sequence():
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    model = nn.Linear(10, 2)
    val_losses = [0.90, 0.70, 0.60, 0.5995, 0.62, 0.61]
    saved_weights_at_epoch_3 = None

    for epoch, loss in enumerate(val_losses, start=1):
        nn.init.uniform_(model.weight)

        if epoch == 3:
            saved_weights_at_epoch_3 = {k: v.clone() for k, v in model.state_dict().items()}

        early_stopping(loss, 0.9, 0.9, model, epoch)

        if early_stopping.early_stop:
            assert epoch == 6, f"Se detuvo prematuramente en la época {epoch}"
            break

    assert early_stopping.early_stop, "No se activó el early stopping"
    assert early_stopping.best_epoch == 3, "No registró la mejor época correctamente"

    model.load_state_dict(early_stopping.best_weights)
    for key in model.state_dict():
        assert torch.equal(model.state_dict()[key], saved_weights_at_epoch_3[key]), (
            "Los pesos restaurados no coinciden"
        )


def test_mlflow_contract(mock_dataset_env, tmp_path, monkeypatch):
    manifest_path, base_dir = mock_dataset_env
    device = torch.device("cpu")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MANIFEST_META_PATH", str(tmp_path / "manifest_meta.json"))
    monkeypatch.setenv("GIT_COMMIT", "mockcommit123")
    monkeypatch.setenv("GIT_DIRTY", "false")

    config = TrainingConfig(
        optimizer="sgd",
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
        patience=2,
        min_delta=0.01,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    loader_args = {
        "manifest_path": manifest_path,
        "base_dir": base_dir,
        "batch_size": 2,
        "image_size": config.image_size,
        "seed_train": config.seed_train,
        "num_workers": 0,
    }
    train_loader = create_dataloader(split="train", **loader_args)
    val_loader = create_dataloader(split="val", **loader_args)

    mock_tags = {"run_kind": "smoke", "grid_row": ""}
    run_id = train_with_mlflow(
        config, model, train_loader, val_loader, optimizer, criterion, device, tags=mock_tags
    )

    run = mlflow.get_run(run_id)
    tags = run.data.tags
    metrics = run.data.metrics
    params = run.data.params

    assert "optimizer" in params
    assert params["optimizer"] == "sgd"
    assert "momentum" in params and params["momentum"] == "0.9"
    assert "seed_aug" in params

    assert tags["release"] == "v0.1.1"
    assert tags["manifest_id"] == "m_123"
    assert tags["git_commit"] == "mockcommit123"
    assert tags["git_dirty"] == "false"
    assert "python_version" in tags
    assert "torch_version" in tags
    assert "torchvision_version" in tags
    assert "platform" in tags
    assert tags["device"] == "cpu"

    assert "best_val_loss" in metrics
    assert "best_val_accuracy" in metrics
    assert "best_val_macro_f1" in metrics
    assert "duration_s" in metrics
    assert "best_epoch" in metrics
    assert "stopped_epoch" in metrics
    assert not any(k.startswith("test_") for k in metrics)

    client = MlflowClient()
    for m in ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1"]:
        history = client.get_metric_history(run_id, m)
        assert len(history) == int(metrics["stopped_epoch"])
        assert history[0].step == 1

    curves_path = download_artifacts(
        run_id=run_id, artifact_path="curves.csv", dst_path=str(tmp_path)
    )
    df = pd.read_csv(curves_path)
    assert len(df) == metrics["stopped_epoch"]
    assert list(df.columns) == [
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]

    ckpt_path = download_artifacts(
        run_id=run_id, artifact_path="checkpoint/best.pt", dst_path=str(tmp_path)
    )
    assert tags["checkpoint_sha256"] == compute_sha256(ckpt_path)


def test_get_git_info_rejects_the_docker_arg_default_commit(monkeypatch):
    """Revisión sobre #48/P3-09: `ARG GIT_COMMIT=unknown` (Dockerfile.ml) es un
    default válido para Docker, no para una corrida -- construir sin pasar
    por `scripts/build_trainer.sh` (p. ej. `docker compose up --build` en un
    clon limpio) no debe registrar "unknown" en MLflow sin que nada lo
    rechace."""
    monkeypatch.setenv("GIT_COMMIT", "unknown")

    with pytest.raises(RuntimeError, match="GIT_COMMIT"):
        get_git_info()


@pytest.mark.parametrize("dirty_value", [None, "", "unknown", "maybe"])
def test_get_git_info_rejects_an_invalid_git_dirty(monkeypatch, dirty_value):
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    if dirty_value is None:
        monkeypatch.delenv("GIT_DIRTY", raising=False)
    else:
        monkeypatch.setenv("GIT_DIRTY", dirty_value)

    with pytest.raises(RuntimeError, match="GIT_DIRTY"):
        get_git_info()


@pytest.mark.parametrize("dirty_value, expected", [("true", "true"), ("FALSE", "false")])
def test_get_git_info_accepts_true_or_false_case_insensitive(monkeypatch, dirty_value, expected):
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    monkeypatch.setenv("GIT_DIRTY", dirty_value)

    commit, dirty = get_git_info()

    assert commit == "abc123"
    assert dirty == expected


def test_mlflow_fails_without_manifest(mock_dataset_env, tmp_path, monkeypatch):
    monkeypatch.setenv("MANIFEST_META_PATH", "ruta_falsa/meta.json")
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
        patience=2,
        min_delta=0.01,
    )
    with pytest.raises(FileNotFoundError):
        train_with_mlflow(
            config, None, None, None, None, None, None, tags={"run_kind": "smoke", "grid_row": ""}
        )


def test_mlflow_fails_without_git(mock_dataset_env, tmp_path, monkeypatch):
    monkeypatch.setenv("MANIFEST_META_PATH", str(tmp_path / "manifest_meta.json"))
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "git")),
    )
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
        patience=2,
        min_delta=0.01,
    )
    with pytest.raises(RuntimeError, match="git_commit"):
        train_with_mlflow(
            config, None, None, None, None, None, None, tags={"run_kind": "smoke", "grid_row": ""}
        )


def test_mlflow_fails_on_nan_val_loss(mock_dataset_env, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MANIFEST_META_PATH", str(tmp_path / "manifest_meta.json"))
    monkeypatch.setenv("GIT_COMMIT", "mock")
    monkeypatch.setenv("GIT_DIRTY", "false")
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
        patience=2,
        min_delta=0.01,
    )

    import training.tracking

    monkeypatch.setattr(training.tracking, "evaluate_epoch", lambda *args: (float("nan"), 0.0, 0.0))
    monkeypatch.setattr(training.tracking, "train_epoch", lambda *args: (0.0, 0.0))

    with pytest.raises(RuntimeError, match="checkpoint"):
        train_with_mlflow(
            config, None, None, None, None, None, None, tags={"run_kind": "smoke", "grid_row": ""}
        )


def test_mlflow_run_fails_on_exception(mock_dataset_env, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MANIFEST_META_PATH", str(tmp_path / "manifest_meta.json"))

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
        patience=2,
        min_delta=0.01,
    )

    mock_tags = {"run_kind": "smoke", "grid_row": ""}
    with pytest.raises(AttributeError):
        train_with_mlflow(config, None, None, None, None, None, None, tags=mock_tags)

    last_run = mlflow.search_runs(experiment_names=["clasificador-perro-gato"]).iloc[0]
    assert last_run["status"] == "FAILED"
