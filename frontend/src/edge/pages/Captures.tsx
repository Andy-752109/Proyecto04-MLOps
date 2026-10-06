import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { MlApiError } from "@/model/api/client";
import { PageHeader } from "@/pipeline/components/PageHeader";
import { getEdgeCaptures } from "../api/captures";
import type { EdgeCapture, EdgeCaptureList } from "../api/contracts";

type CapturesState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: EdgeCaptureList };

function captureErrorMessage(error: unknown): string {
  if (error instanceof MlApiError) {
    if (error.status === 0)
      return "No se pudo conectar con ml-api. Comprueba la conexión e inténtalo de nuevo.";
    if (error.status === 503) {
      return "El servicio de capturas o AWS no está disponible temporalmente. Inténtalo de nuevo.";
    }
    return "ml-api no pudo cargar las capturas. Inténtalo de nuevo.";
  }
  if (
    error instanceof SyntaxError ||
    (error instanceof Error && error.message.startsWith("Respuesta inválida de ml-api"))
  ) {
    return "ml-api devolvió una respuesta inválida o incompatible con el contrato de capturas.";
  }
  return "No se pudieron cargar las capturas. Inténtalo de nuevo.";
}

function newestFirst(items: EdgeCapture[]): EdgeCapture[] {
  return [...items].sort(
    (a, b) =>
      Date.parse(b.captured_at) - Date.parse(a.captured_at) ||
      a.capture_id.localeCompare(b.capture_id)
  );
}

function CaptureCard({ capture }: { capture: EdgeCapture }) {
  const [imageFailed, setImageFailed] = useState(false);

  return (
    <article className="overflow-hidden rounded-2xl border border-border bg-surface">
      <div className="aspect-video bg-sidebar">
        {imageFailed ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-5 text-center text-sm text-ink-muted">
            <p>No se pudo cargar la imagen; la URL puede haber expirado.</p>
            <p>Pulsa Actualizar para solicitar una URL nueva.</p>
          </div>
        ) : (
          <img
            src={capture.image_url}
            alt={`Frame completo de la captura ${capture.capture_id}`}
            className="h-full w-full object-contain"
            referrerPolicy="no-referrer"
            onError={() => setImageFailed(true)}
          />
        )}
      </div>
      <div className="space-y-3 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-base font-semibold text-ink">{capture.predicted_class}</h2>
          <span className="rounded-full bg-accent-lilac-soft px-2.5 py-1 text-sm font-medium text-accent-lilac">
            {new Intl.NumberFormat("es-MX", {
              style: "percent",
              maximumFractionDigits: 1,
            }).format(capture.confidence)}
          </span>
        </div>
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-sm">
          <dt className="text-ink-muted">Capturada</dt>
          <dd>
            <time dateTime={capture.captured_at}>{capture.captured_at}</time>
          </dd>
          <dt className="text-ink-muted">Recibida</dt>
          <dd>
            <time dateTime={capture.received_at}>{capture.received_at}</time>
          </dd>
          <dt className="text-ink-muted">capture_id</dt>
          <dd className="break-all font-mono text-xs">{capture.capture_id}</dd>
          <dt className="text-ink-muted">device_id</dt>
          <dd className="break-all">{capture.device_id}</dd>
          <dt className="text-ink-muted">model_version</dt>
          <dd className="break-all">{capture.model_version}</dd>
          <dt className="text-ink-muted">Recorte</dt>
          <dd>
            {capture.crop === null
              ? "Sin recorte"
              : `x=${capture.crop.x}, y=${capture.crop.y}, ancho=${capture.crop.width}, alto=${capture.crop.height}; frame=${capture.crop.frame_width}×${capture.crop.frame_height}`}
          </dd>
        </dl>
      </div>
    </article>
  );
}

export function CapturesPage() {
  const [state, setState] = useState<CapturesState>({ status: "loading" });
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState({ status: "loading" });
    getEdgeCaptures(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setState({ status: "success", data });
      })
      .catch((error: unknown) => {
        // Solo mensajes propios: no exponer respuestas internas ni URLs prefirmadas.
        if (!controller.signal.aborted) {
          setState({ status: "error", message: captureErrorMessage(error) });
        }
      });
  }, []);

  useEffect(() => {
    refresh();
    return () => controllerRef.current?.abort();
  }, [refresh]);

  return (
    <main className="flex flex-1 flex-col gap-6 p-6 lg:p-8">
      <PageHeader
        title="Capturas Edge"
        subtitle="Capturas enviadas por dispositivos edge, de la más reciente a la más antigua."
      >
        <Button type="button" onClick={refresh}>
          Actualizar
        </Button>
      </PageHeader>

      {state.status === "loading" && (
        <div
          role="status"
          aria-label="Cargando capturas edge"
          className="grid gap-4 md:grid-cols-2"
        >
          <Skeleton className="h-80" />
          <Skeleton className="h-80" />
        </div>
      )}
      {state.status === "error" && (
        <ErrorState
          title="No se pudieron cargar las capturas edge."
          message={state.message}
          onRetry={refresh}
        />
      )}
      {state.status === "success" &&
        (state.data.items.length === 0 ? (
          <EmptyState
            hasFilters={false}
            title="No hay capturas edge"
            description="Cuando lleguen capturas, aparecerán aquí. Pulsa Actualizar para consultar de nuevo."
          />
        ) : (
          <section className="space-y-4">
            <p className="text-sm text-ink-muted">
              {state.data.total} capturas válidas · bucket {state.data.bucket}
            </p>
            <ul aria-label="Capturas Edge" className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {newestFirst(state.data.items).map((capture) => (
                <li key={capture.capture_id}>
                  <CaptureCard capture={capture} />
                </li>
              ))}
            </ul>
          </section>
        ))}
    </main>
  );
}
