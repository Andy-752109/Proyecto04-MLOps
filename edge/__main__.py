"""CLI del edge, desde la raíz del repositorio:

python -m edge run [--config edge/config.yaml] [--mode manual|interval] [--count N] [--bucket B]
python -m edge status [--config edge/config.yaml]
python -m edge retry <capture_id>... | --pending [--bucket B]
python -m edge export [--out DIR]
python -m edge cameras
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from edge.backend import create_backend, sha256_file
from edge.camera import Camera, CameraError, list_camera_names
from edge.config import ConfigError, EdgeConfig, load_config
from edge.event_validator import EventValidationError
from edge.export import export
from edge.pipeline import Capture, EdgeApp, ModelVerificationError, now_utc, read_log
from edge.sync import (
    STATUSES,
    BackgroundSender,
    UnknownCaptureError,
    UploadLog,
    retry,
    retry_ids,
    status_of,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
# Al salir se esperan los envíos en cola; sin red cada uno tarda ~10 s en fallar.
SHUTDOWN_TIMEOUT_S = 60
log = logging.getLogger("edge")


def setup_logging(config: EdgeConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    for handler in (
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(config.data_dir / "edge.log", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def camera_label(config: EdgeConfig) -> str:
    camera = config.camera
    return camera.name if camera.name is not None else f"índice {camera.index}"


def cameras() -> int:
    try:
        names = list_camera_names()
    except CameraError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not names:
        print("no se encontraron cámaras (la lista por nombre solo funciona en Windows)")
        return 1
    for index, name in enumerate(names):
        print(f"{index}: {name}")
    return 0


def show(capture: Capture, number: int) -> None:
    event = capture.event
    print(
        f"[{number}] {event['predicted_class'].upper():3s} {event['confidence']:.3f}"
        f" | pre {event['preprocess_ms']:.1f} ms | inf {event['inference_ms']:.1f} ms"
        f" | {event['capture_id']}",
        flush=True,
    )


def capture_once(
    app: EdgeApp, camera: Camera, number: int, sender: BackgroundSender | None = None
) -> None:
    frame = camera.read()
    captured_at = now_utc()
    try:
        capture = app.process(frame, captured_at)
    except (EventValidationError, ValueError) as exc:
        # Una captura mala no tumba la app: se registra y se sigue.
        log.error("captura descartada: %s", exc)
        return
    log.info(
        "captura %s %s %.4f",
        capture.event["capture_id"],
        capture.event["predicted_class"],
        capture.event["confidence"],
    )
    show(capture, number)
    if sender is not None:
        # El envío corre en otro hilo: la siguiente captura no lo espera.
        sender.submit(capture.event, capture.image_path)


def run(config: EdgeConfig, mode: str, count: int | None) -> int:
    app = EdgeApp(config, create_backend(config.model.runtime, config.model.threads))
    app.start()
    log.info("modo %s, dispositivo %s, cámara %s", mode, config.device_id, camera_label(config))
    sender = None
    if config.aws.enabled:
        sender = BackgroundSender(config.aws, UploadLog(config.upload_log_path)).start()
        log.info("envío a s3://%s (perfil %s)", config.aws.bucket, config.aws.profile or "default")
    else:
        log.warning("sin aws.bucket: las capturas solo se guardan en local")
    number = 0
    try:
        with Camera(config.camera) as camera:
            try:
                while count is None or number < count:
                    if mode == "manual":
                        answer = input("Enter = capturar, q = salir > ").strip().lower()
                        if answer == "q":
                            break
                    elif number:
                        time.sleep(config.interval_seconds)
                    number += 1
                    capture_once(app, camera, number, sender)
            except (KeyboardInterrupt, EOFError):
                print()
    finally:
        if sender is not None:
            waiting = sender.submitted - len(sender.results)
            if waiting:
                log.info("esperando %d envíos en cola (máx. %d s)", waiting, SHUTDOWN_TIMEOUT_S)
            left = sender.close(SHUTDOWN_TIMEOUT_S)
            if left:
                log.warning(
                    "%d envíos quedaron pending; reenvía con: python -m edge retry --pending", left
                )
    log.info("fin: %d capturas en esta sesión", number)
    return 0


def retry_command(config: EdgeConfig, capture_ids: list[str], pending: bool) -> int:
    if not config.aws.enabled:
        log.error("falta aws.bucket en la config (o --bucket)")
        return 2
    try:
        ids = retry_ids(config, pending, capture_ids)
    except UnknownCaptureError as exc:
        log.error("%s", exc.args[0])
        return 2
    if not ids:
        print("no hay capturas pendientes de envío")
        return 0
    results = retry(config, ids)
    for result in results:
        print(
            f"{result.capture_id} {result.status} {result.upload_ms:.0f} ms"
            + (f" | {result.error}" if result.error else ""),
            flush=True,
        )
    counts = Counter(r.status for r in results)
    print("resumen: " + ", ".join(f"{s} {counts[s]}" for s in STATUSES if counts[s]))
    return 1 if counts["failed"] else 0


def export_command(config: EdgeConfig, out_dir: Path | None) -> int:
    json_path, csv_path, rows = export(config, out_dir or config.data_dir / "export")
    print(f"{rows} capturas -> {json_path} y {csv_path}")
    return 0


def status(config: EdgeConfig) -> int:
    lines = read_log(config.log_path)
    model = config.model.path
    print(f"dispositivo : {config.device_id}")
    print(f"modo        : {config.mode} (intervalo {config.interval_seconds:g} s)")
    print(f"cámara      : {camera_label(config)}")
    print(f"recorte     : {config.crop}")
    print(f"modelo      : {model} ({config.model.version}, {config.model.runtime})")
    if model.is_file():
        ok = sha256_file(model) == config.model.sha256
        print(f"sha256      : {'OK' if ok else 'NO COINCIDE'}")
    else:
        print("sha256      : sin modelo en caché")
    print(f"capturas    : {len(lines)} en {config.log_path}")
    print(f"bucket      : {config.aws.bucket or '(sin envío)'}")
    latest = UploadLog(config.upload_log_path).latest()
    counts = Counter(status_of(latest, line["event"]["capture_id"]) for line in lines)
    print("envíos      : " + ", ".join(f"{s} {counts[s]}" for s in STATUSES))
    if lines:
        last = lines[-1]["event"]
        print(
            f"última      : {last['captured_at']} {last['predicted_class']}"
            f" {last['confidence']:.3f} ({last['capture_id']})"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m edge", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="captura y clasifica")
    run_parser.add_argument("--mode", choices=("manual", "interval"))
    run_parser.add_argument("--count", type=int, help="termina tras N capturas")
    run_parser.add_argument(
        "--bucket", help="sustituye aws.bucket (p. ej. para provocar una falla)"
    )
    sub.add_parser("status", help="configuración, modelo, log local y estado de envíos")
    retry_parser = sub.add_parser("retry", help="reenvía el mismo evento de capturas ya hechas")
    retry_parser.add_argument("capture_ids", nargs="*", metavar="capture_id")
    retry_parser.add_argument(
        "--pending", action="store_true", help="todas las que no están sent/already_sent"
    )
    retry_parser.add_argument("--bucket", help="sustituye aws.bucket")
    export_parser = sub.add_parser("export", help="eventos locales a CSV y JSON")
    export_parser.add_argument("--out", type=Path, help="carpeta de salida (data_dir/export)")
    sub.add_parser("cameras", help="lista las cámaras con su índice actual (Windows)")
    args = parser.parse_args(argv)
    if args.command == "retry" and bool(args.capture_ids) == args.pending:
        parser.error("retry necesita capture_ids o --pending (no ambos)")

    if args.command == "cameras":
        return cameras()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config inválida: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "bucket", None):
        config = dataclasses.replace(
            config, aws=dataclasses.replace(config.aws, bucket=args.bucket.strip())
        )
    if args.command == "status":
        return status(config)
    if args.command == "export":
        return export_command(config, args.out)

    setup_logging(config)
    if args.command == "retry":
        return retry_command(config, args.capture_ids, args.pending)
    try:
        return run(config, args.mode or config.mode, args.count)
    except (ModelVerificationError, CameraError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
