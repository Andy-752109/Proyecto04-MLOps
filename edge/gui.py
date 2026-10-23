"""Interfaz gráfica del edge (`python -m edge gui`): vista en vivo y botón para capturar.

Hace lo mismo que `python -m edge run`: cada foto pasa por `EdgeApp.process` (recorte,
preprocesamiento, inferencia, evento validado, `images/`, `crops/`, `captures.jsonl`) y por
`BackgroundSender` (envío a S3 en segundo plano). Solo cambia cómo se pide la foto: en lugar
de Enter hay un botón, y antes de apretarlo se ve hacia dónde apunta la cámara.

La vista y la captura comparten un solo flujo (`FrameGrabber`): la foto es el último cuadro
que se ve en pantalla. Tkinter solo se toca desde el hilo principal; el modelo y las capturas
corren en un hilo aparte y devuelven sus resultados por una cola.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk
from typing import Any

from PIL import Image, ImageTk

from edge.backend import CLASSES, create_backend
from edge.camera import Camera, ImageSource
from edge.config import EdgeConfig
from edge.event_validator import EventValidationError
from edge.pipeline import Capture, EdgeApp, ModelVerificationError, now_utc
from edge.preview import Frame, FrameGrabber, render_preview
from edge.sync import DONE, SHUTDOWN_TIMEOUT_S, BackgroundSender, UploadLog

log = logging.getLogger("edge")

REFRESH_MS = 33  # ~30 imágenes por segundo
UPLOADS_EVERY_S = 0.5
STALE_FRAME_S = 1.0  # sin cuadros nuevos por más de esto, la cámara se da por detenida
THUMBNAIL_PX = 200
UPLOAD_TEXT = {
    "pending": "enviando…",
    "sent": "enviada",
    "already_sent": "ya enviada",
    "failed": "FALLÓ",
}
ERROR_COLOR = "#b3261e"
SSH_NOTE = (
    "Esta sesión es por SSH: la ventana se abre en la pantalla de la laptop, no en tu terminal, "
    "y desde SSH Windows normalmente no la muestra ni deja usar la cámara. Abre la GUI desde la "
    "sesión de la laptop (frente a ella o por escritorio remoto); por SSH usa `python -m edge run`."
)
NO_DISPLAY_NOTE = "La GUI necesita una pantalla. Sin ella usa `python -m edge run`."
CAMERA_HINT = (
    "Revisa que la cámara esté conectada y que ninguna otra aplicación la esté usando.\n"
    "`python -m edge cameras` lista las disponibles."
)


def local_time(captured_at: str) -> str:
    """`HH:MM:SS` en la zona del sistema, a partir de `captured_at` del evento."""
    return f"{datetime.fromisoformat(captured_at).astimezone():%H:%M:%S}"


class EdgeGui:
    def __init__(
        self,
        root: tk.Tk,
        config: EdgeConfig,
        app: EdgeApp,
        grabber: FrameGrabber,
        sender: BackgroundSender | None,
        source_label: str,
        source_hint: str | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.app = app
        self.grabber = grabber
        self.sender = sender
        self.source_label = source_label
        self.source_hint = source_hint
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edge-gui")
        self._model_ready = False
        self._model_error: str | None = None
        self._busy = False
        self._count = 0
        self._shown: dict[str, str] = {}  # capture_id -> estado de envío que muestra la tabla
        self._drawn: tuple[int, int, int, bool] | None = None
        self._uploads_at = 0.0
        self._next_auto = 0.0
        self._closing = False
        self._finished = False
        self._tick_failed = False
        self._camera_error: str | None = None  # el último que se anunció en el mensaje
        self._left: int | None = None  # envíos sin intentar al cerrar; None = aún esperando
        self._photo: ImageTk.PhotoImage | None = None  # Tk no guarda la referencia
        self._thumbnail: ImageTk.PhotoImage | None = None
        self._build()
        self._pool.submit(self._load_model)
        self._poll_job: str | None = None  # espera de envíos al cerrar
        self._job = root.after(REFRESH_MS, self._tick)

    # ---- ventana ----

    def _build(self) -> None:
        root, config = self.root, self.config
        root.title(f"Edge · {config.device_id}")
        root.minsize(960, 600)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.bind("<space>", self._on_key)
        root.bind("<Return>", self._on_key)

        style = ttk.Style(root)
        base = tkfont.nametofont("TkDefaultFont")
        self._message_font = base.copy()
        self._message_font.configure(size=base.cget("size") + 3)
        self._big_font = base.copy()
        self._big_font.configure(size=30, weight="bold")
        style.configure("Capture.TButton", font=(base.cget("family"), 13, "bold"))
        style.configure("Error.TLabel", foreground=ERROR_COLOR)

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        self._build_live(outer).grid(row=0, column=0, sticky="nsew")
        self._build_side(outer).grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.message = tk.StringVar(value="Cargando el modelo…")
        self.message_label = ttk.Label(outer, textvariable=self.message, anchor="w")
        self.message_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_live(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(frame, width=640, height=480, highlightthickness=0)
        self.canvas.configure(background="#15171a")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._image_item = self.canvas.create_image(0, 0, anchor="center")
        self._text_item = self.canvas.create_text(
            0, 0, fill="#c9ced6", justify="center", font=self._message_font
        )

        controls = ttk.Frame(frame)
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.capture_button = ttk.Button(
            controls,
            text="Capturar foto  (Espacio)",
            style="Capture.TButton",
            command=self.capture,
            takefocus=False,
        )
        self.capture_button.grid(row=0, column=0, rowspan=2, sticky="ns", ipadx=14, ipady=8)
        self.capture_button.state(["disabled"])

        options = ttk.Frame(controls)
        options.grid(row=0, column=1, rowspan=2, sticky="w", padx=(18, 0))
        self.show_crop = tk.BooleanVar(value=True)
        if self.config.crop is not None:
            crop_option: ttk.Widget = ttk.Checkbutton(
                options,
                text="Marcar la zona que ve el modelo",
                variable=self.show_crop,
                takefocus=False,
            )
        else:
            crop_option = ttk.Label(options, text="Sin recorte: el modelo ve todo el cuadro")
        crop_option.grid(row=0, column=0, columnspan=4, sticky="w")

        self.auto = tk.BooleanVar(value=False)
        self.interval_var = tk.StringVar(value=f"{self.config.interval_seconds:g}")
        self.countdown = tk.StringVar()
        ttk.Checkbutton(
            options,
            text="Captura automática cada",
            variable=self.auto,
            command=self._on_auto_toggle,
            takefocus=False,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.interval_box = ttk.Spinbox(
            options, from_=1, to=3600, width=5, textvariable=self.interval_var
        )
        self.interval_box.grid(row=1, column=1, padx=(6, 4), pady=(6, 0))
        ttk.Label(options, text="s").grid(row=1, column=2, pady=(6, 0))
        ttk.Label(options, textvariable=self.countdown).grid(
            row=1, column=3, padx=(10, 0), pady=(6, 0)
        )
        return frame

    def _build_side(self, parent: ttk.Frame) -> ttk.Frame:
        side = ttk.Frame(parent)
        side.columnconfigure(0, weight=1, minsize=340)
        side.rowconfigure(2, weight=1)

        info = ttk.LabelFrame(side, text="Equipo", padding=8)
        info.grid(row=0, column=0, sticky="ew")
        info.columnconfigure(0, weight=1)
        self.camera_text = tk.StringVar()
        self.model_text = tk.StringVar(value="cargando…")
        self.upload_text = tk.StringVar(value=self._upload_summary())
        self.upload_label = ttk.Label(info, textvariable=self.upload_text, wraplength=320)
        ttk.Label(info, text=f"Dispositivo: {self.config.device_id}").grid(sticky="w")
        ttk.Label(info, text=f"Fuente: {self.source_label}", wraplength=320).grid(sticky="w")
        ttk.Label(info, textvariable=self.camera_text, wraplength=320).grid(sticky="w")
        ttk.Label(
            info,
            text=f"Modelo: {self.config.model.version} · {self.app.backend.name}",
            wraplength=320,
        ).grid(sticky="w")
        ttk.Label(info, textvariable=self.model_text, wraplength=320).grid(sticky="w")
        self.upload_label.grid(sticky="w")

        last = ttk.LabelFrame(side, text="Última captura", padding=8)
        last.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        last.columnconfigure(1, weight=1)
        self.thumbnail_label = ttk.Label(last, anchor="center", text="(aún no hay capturas)")
        self.thumbnail_label.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.class_label = ttk.Label(last, text="—", font=self._big_font, anchor="center")
        self.class_label.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.bars: dict[str, tuple[ttk.Progressbar, ttk.Label]] = {}
        for row, name in enumerate(CLASSES, start=2):
            bar = ttk.Progressbar(last, maximum=1.0)
            percent = ttk.Label(last, text="", width=7, anchor="e")
            ttk.Label(last, text=name, width=4).grid(row=row, column=0, sticky="w")
            bar.grid(row=row, column=1, sticky="ew", padx=6)
            percent.grid(row=row, column=2, sticky="e")
            self.bars[name] = (bar, percent)
        self.detail_label = ttk.Label(last, text="", wraplength=320, justify="left")
        self.detail_label.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.id_var = tk.StringVar()
        # Solo lectura pero seleccionable: el capture_id se copia para buscarlo en el portal.
        ttk.Entry(last, textvariable=self.id_var, state="readonly").grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )

        history = ttk.LabelFrame(side, text="Esta sesión", padding=4)
        history.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        history.columnconfigure(0, weight=1)
        history.rowconfigure(0, weight=1)
        columns = (
            ("n", "#", 30, "e"),
            ("hora", "Hora", 68, "center"),
            ("clase", "Clase", 46, "center"),
            ("conf", "Conf.", 52, "e"),
            ("envio", "Envío", 90, "w"),
        )
        self.tree = ttk.Treeview(
            history, columns=[c[0] for c in columns], show="headings", height=6
        )
        for name, title, width, anchor in columns:
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, anchor=anchor, stretch=name == "envio")
        self.tree.tag_configure("failed", foreground=ERROR_COLOR)
        scroll = ttk.Scrollbar(history, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        return side

    def _say(self, text: str, error: bool = False) -> None:
        self.message.set(text)
        self.message_label.configure(style="Error.TLabel" if error else "TLabel")

    # ---- trabajo en el hilo aparte ----

    def _load_model(self) -> None:
        try:
            self.app.start()
        except Exception as error:  # el mensaje llega a la ventana, que no tiene consola
            log.error("%s", error)
            named = isinstance(error, ModelVerificationError)
            text = str(error) if named else f"{type(error).__name__}: {error}"
            self._events.put(("model_error", text))
        else:
            self._events.put(("model_ready", None))

    def _process(self, frame: Frame, captured_at: datetime) -> None:
        try:
            capture = self.app.process(frame.image, captured_at)
        except (EventValidationError, ValueError) as error:
            # Una captura mala no tumba la app: se registra y se sigue.
            log.error("captura descartada: %s", error)
            self._events.put(("capture_failed", str(error)))
            return
        except Exception as error:  # p. ej. disco lleno: la ventana sigue viva
            log.exception("captura fallida")
            self._events.put(("capture_failed", f"{type(error).__name__}: {error}"))
            return
        event = capture.event
        log.info(
            "captura %s %s %.4f",
            event["capture_id"],
            event["predicted_class"],
            event["confidence"],
        )
        if self.sender is not None:
            # El envío corre en otro hilo: la siguiente captura no lo espera.
            self.sender.submit(event, capture.image_path)
        self._events.put(("capture", capture))

    # ---- captura ----

    def _blocked_reason(self, frame: Frame | None) -> str | None:
        if self._closing:
            return "la ventana se está cerrando"
        if self.grabber.error:
            return "la cámara no está disponible"
        if frame is None:
            return "la cámara aún no entrega imagen"
        if time.monotonic() - frame.at > STALE_FRAME_S:
            return "la cámara dejó de entregar cuadros"
        if self._model_error:
            return "el modelo no se pudo cargar"
        if not self._model_ready:
            return "el modelo se está cargando"
        if self._busy:
            return "la captura anterior sigue en proceso"
        return None

    def capture(self, announce: bool = True) -> None:
        """Toma el último cuadro de la vista y lo clasifica en el hilo aparte."""
        frame = self.grabber.latest()
        reason = self._blocked_reason(frame)
        if reason is not None or frame is None:
            if announce:
                self._say(f"No se puede capturar: {reason}.", error=True)
            return
        self._busy = True
        self.capture_button.state(["disabled"])
        self._pool.submit(self._process, frame, now_utc())

    def _on_key(self, event: tk.Event) -> None:
        # En el campo de segundos (o en el id) Espacio/Enter son del campo, no de la captura.
        if not isinstance(event.widget, ttk.Entry):
            self.capture()

    def _interval(self) -> float:
        try:
            seconds = float(self.interval_var.get())
        except ValueError:
            return self.config.interval_seconds
        return seconds if seconds > 0 else self.config.interval_seconds

    def _on_auto_toggle(self) -> None:
        if self.auto.get():
            self._next_auto = time.monotonic() + self._interval()
            self._say(f"Captura automática activada: una foto cada {self._interval():g} s.")
        else:
            self._say("Captura automática desactivada.")

    # ---- resultados ----

    def _drain_events(self) -> None:
        handlers = {
            "model_ready": self._on_model_ready,
            "model_error": self._on_model_error,
            "capture": self._on_capture,
            "capture_failed": self._on_capture_failed,
        }
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                return
            handlers[kind](payload)

    def _on_model_ready(self, _: None) -> None:
        self._model_ready = True
        self.model_text.set("verificado y listo")
        if not self.grabber.error:  # con la cámara caída, el mensaje ya dice por qué no hay imagen
            self._say("Listo. Apunta la cámara y presiona Espacio o el botón para capturar.")

    def _on_model_error(self, text: str) -> None:
        self._model_error = text
        self.model_text.set("NO disponible")
        self._say(f"Modelo: {text}", error=True)

    def _on_capture_failed(self, text: str) -> None:
        self._busy = False
        self._say(f"Captura descartada: {text}", error=True)

    def _on_capture(self, capture: Capture) -> None:
        self._busy = False
        self._count += 1
        event = capture.event
        if self.sender is not None:
            self._shown[event["capture_id"]] = "pending"
        self.tree.insert(
            "",
            0,
            iid=event["capture_id"],
            values=(
                self._count,
                local_time(event["captured_at"]),
                event["predicted_class"].upper(),
                f"{event['confidence']:.3f}",
                UPLOAD_TEXT["pending"] if self.sender is not None else "solo local",
            ),
        )
        self._show_result(capture)
        self._say(
            f"Captura {self._count} guardada: {event['predicted_class'].upper()}"
            f" {event['confidence']:.3f} ({event['capture_id']})"
        )

    def _show_result(self, capture: Capture) -> None:
        event = capture.event
        self.class_label.configure(text=event["predicted_class"].upper())
        for name, (bar, percent) in self.bars.items():
            probability = event["probabilities"][name]
            bar.configure(value=probability)
            percent.configure(text=f"{probability:.1%}")
        self.detail_label.configure(
            text=(
                f"confianza {event['confidence']:.3f} · {local_time(event['captured_at'])}\n"
                f"preproceso {event['preprocess_ms']:.1f} ms · "
                f"inferencia {event['inference_ms']:.1f} ms"
            )
        )
        self.id_var.set(event["capture_id"])
        try:
            # Lo que vio el modelo: el recorte si lo hay y, si no, el cuadro completo.
            with Image.open(capture.crop_path or capture.image_path) as picture:
                picture.thumbnail((THUMBNAIL_PX, THUMBNAIL_PX))
                self._thumbnail = ImageTk.PhotoImage(picture)
        except OSError as error:  # la miniatura es de cortesía: el resultado ya se mostró
            log.warning("no se pudo mostrar la miniatura: %s", error)
        else:
            self.thumbnail_label.configure(image=self._thumbnail, text="")

    # ---- envío ----

    def _upload_summary(self) -> str:
        if self.sender is None:
            return "Envío: solo local (aws.bucket vacío)"
        results = list(self.sender.results)
        sent = sum(r.status in DONE for r in results)
        failed = sum(r.status == "failed" for r in results)
        waiting = self.sender.submitted - len(results)
        text = f"Envío: s3://{self.config.aws.bucket}"
        if sent:
            text += f" · {sent} enviadas"
        if waiting:
            text += f" · {waiting} en cola"
        if failed:
            text += f"\n{failed} FALLARON: python -m edge retry --pending"
        return text

    def _update_uploads(self) -> None:
        now = time.monotonic()
        if self.sender is None or now < self._uploads_at:
            return
        self._uploads_at = now + UPLOADS_EVERY_S
        done = {r.capture_id: r.status for r in list(self.sender.results)}
        for capture_id, shown in list(self._shown.items()):
            status = done.get(capture_id, "pending")
            if status != shown:
                self._shown[capture_id] = status
                self.tree.set(capture_id, "envio", UPLOAD_TEXT[status])
                self.tree.item(capture_id, tags=("failed",) if status == "failed" else ())
        text = self._upload_summary()
        self.upload_text.set(text)
        self.upload_label.configure(style="Error.TLabel" if "FALLARON" in text else "TLabel")

    # ---- ciclo de refresco ----

    def _tick(self) -> None:
        try:
            self._drain_events()
            frame = self.grabber.latest()
            self._update_source(frame)
            self._draw(frame)
            self._auto_capture()
            self._update_uploads()
        except Exception:
            # Si esto escapara, Tk dejaría de llamarlo y la ventana quedaría congelada.
            if not self._tick_failed:
                self._tick_failed = True
                log.exception("error al refrescar la ventana")
        finally:
            if not self._finished:
                self._job = self.root.after(REFRESH_MS, self._tick)

    def _update_source(self, frame: Frame | None) -> None:
        error = self.grabber.error
        if error:
            self.camera_text.set(f"Sin imagen: {error}")
            if error != self._camera_error:
                self._camera_error = error
                self._say(f"Sin imagen: {error}", error=True)
        elif frame is None:
            self.camera_text.set(f"Abriendo {self.source_label}…")
        else:
            height, width = frame.image.shape[:2]
            self.camera_text.set(f"En vivo · {width}x{height} · {self.grabber.fps:.0f} fps")
        ready = self._blocked_reason(frame) is None
        disabled = self.capture_button.instate(["disabled"])
        if ready and disabled:
            self.capture_button.state(["!disabled"])
        elif not ready and not disabled:
            self.capture_button.state(["disabled"])

    def _draw(self, frame: Frame | None) -> None:
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        self.canvas.coords(self._image_item, width / 2, height / 2)
        self.canvas.coords(self._text_item, width / 2, height / 2)
        error = self.grabber.error
        if frame is None or error:
            # Cámara que no abre o que se perdió: el motivo va en la vista, no en una consola.
            hint = f"\n\n{self.source_hint}" if error and self.source_hint else ""
            message = error or f"Abriendo {self.source_label}…"
            self.canvas.itemconfigure(self._image_item, state="hidden")
            self.canvas.itemconfigure(
                self._text_item, text=message + hint, width=max(120, width - 80), state="normal"
            )
            self._drawn = None
            return
        key = (frame.number, width, height, self.show_crop.get())
        if key == self._drawn:
            return
        self._drawn = key
        image = render_preview(frame.image, (width, height), self.config.crop, key[3])
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self._image_item, image=self._photo, state="normal")
        self.canvas.itemconfigure(self._text_item, state="hidden")

    def _auto_capture(self) -> None:
        if not self.auto.get():
            if self.countdown.get():
                self.countdown.set("")
            return
        now = time.monotonic()
        if now >= self._next_auto:
            self._next_auto = now + self._interval()
            self.capture(announce=False)
        remaining = f"próxima en {max(0.0, self._next_auto - now):.0f} s"
        if remaining != self.countdown.get():
            self.countdown.set(remaining)

    # ---- cierre ----

    def _close(self) -> None:
        """Cierra la ventana esperando los envíos en cola; un segundo clic sale sin esperar."""
        if self._closing:
            self._finish()
            return
        self._closing = True
        self.auto.set(False)
        self.capture_button.state(["disabled"])
        self.grabber.stop()
        self._pool.shutdown(wait=True)  # deja terminar la captura en curso
        sender = self.sender
        waiting = 0 if sender is None else sender.submitted - len(sender.results)
        if sender is None or not waiting:
            self._finish()
            return
        log.info("esperando %d envíos en cola (máx. %d s)", waiting, SHUTDOWN_TIMEOUT_S)
        self._say(
            f"Esperando {waiting} envíos en cola… (cierra otra vez para salir sin esperarlos)"
        )
        threading.Thread(
            target=self._wait_sender, args=(sender,), name="edge-closer", daemon=True
        ).start()
        self._poll_job = self.root.after(100, self._poll_closer)

    def _wait_sender(self, sender: BackgroundSender) -> None:
        self._left = sender.close(SHUTDOWN_TIMEOUT_S)

    def _poll_closer(self) -> None:
        if self._finished:
            return
        if self._left is None:
            self._poll_job = self.root.after(100, self._poll_closer)
            return
        if self._left:
            log.warning(
                "%d envíos quedaron pending; reenvía con: python -m edge retry --pending",
                self._left,
            )
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        # Un temporizador que sobreviva al destroy() imprime un error de Tcl.
        for job in (self._job, self._poll_job):
            if job is not None:
                self.root.after_cancel(job)
        log.info("fin: %d capturas en esta sesión", self._count)
        self.root.destroy()


def over_ssh() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))


def run_gui(config: EdgeConfig, image: Path | None = None) -> int:
    """Abre la ventana. Con `image` la fuente es ese archivo en lugar de la cámara."""
    try:
        root = tk.Tk()
    except tk.TclError as error:  # sin pantalla (p. ej. SSH sin DISPLAY)
        log.error("no se pudo abrir la ventana: %s", error)
        log.error("%s", SSH_NOTE if over_ssh() else NO_DISPLAY_NOTE)
        return 1
    if over_ssh():
        log.warning("%s", SSH_NOTE)
    camera = config.camera
    if image is not None:
        label, hint, open_source = f"imagen {image}", None, partial(ImageSource, image)
    else:
        name = camera.name if camera.name is not None else f"índice {camera.index}"
        label, hint, open_source = f"cámara {name}", CAMERA_HINT, partial(Camera, camera)

    app = EdgeApp(config, create_backend(config.model.runtime, config.model.threads))
    sender = None
    if config.aws.enabled:
        sender = BackgroundSender(config.aws, UploadLog(config.upload_log_path)).start()
        log.info("envío a s3://%s (perfil %s)", config.aws.bucket, config.aws.profile or "default")
    else:
        log.warning("sin aws.bucket: las capturas solo se guardan en local")
    log.info("modo gui, dispositivo %s, %s", config.device_id, label)
    grabber = FrameGrabber(open_source).start()
    EdgeGui(root, config, app, grabber, sender, label, hint)
    root.mainloop()
    return 0
