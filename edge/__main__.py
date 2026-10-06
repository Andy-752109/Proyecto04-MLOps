"""CLI del edge, desde la raíz del repositorio:

python -m edge run [--config edge/config.yaml] [--mode manual|interval] [--count N]
python -m edge status [--config edge/config.yaml]
python -m edge cameras
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from edge.backend import create_backend, sha256_file
from edge.camera import Camera, CameraError, list_camera_names
from edge.config import ConfigError, EdgeConfig, load_config
from edge.event_validator import EventValidationError
from edge.pipeline import Capture, EdgeApp, ModelVerificationError, now_utc, read_log

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
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


def capture_once(app: EdgeApp, camera: Camera, number: int) -> None:
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


def run(config: EdgeConfig, mode: str, count: int | None) -> int:
    app = EdgeApp(config, create_backend(config.model.runtime, config.model.threads))
    app.start()
    log.info("modo %s, dispositivo %s, cámara %s", mode, config.device_id, camera_label(config))
    with Camera(config.camera) as camera:
        number = 0
        try:
            while count is None or number < count:
                if mode == "manual":
                    answer = input("Enter = capturar, q = salir > ").strip().lower()
                    if answer == "q":
                        break
                elif number:
                    time.sleep(config.interval_seconds)
                number += 1
                capture_once(app, camera, number)
        except (KeyboardInterrupt, EOFError):
            print()
    log.info("fin: %d capturas en esta sesión", number)
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
    sub.add_parser("status", help="configuración, modelo y log local")
    sub.add_parser("cameras", help="lista las cámaras con su índice actual (Windows)")
    args = parser.parse_args(argv)

    if args.command == "cameras":
        return cameras()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config inválida: {exc}", file=sys.stderr)
        return 2
    if args.command == "status":
        return status(config)

    setup_logging(config)
    try:
        return run(config, args.mode or config.mode, args.count)
    except (ModelVerificationError, CameraError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
