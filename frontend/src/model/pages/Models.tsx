import { type ReactNode, useCallback, useEffect, useState } from "react";
import { PageHeader } from "@/pipeline/components/PageHeader";
import { MlApiError } from "../api/client";
import type { ModelDetail, ModelListResponse, ModelSummary } from "../api/contracts";
import { getModel, getModels, setActiveModel } from "../api/models";

type ListState =
  | { status: "loading" }
  | { status: "error" }
  | { status: "success"; data: ModelListResponse };

type DetailState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: ModelDetail };

const TAG_CLASSES = {
  active: "bg-status-done-soft text-status-done",
  selected: "bg-accent-lilac-soft text-accent-lilac",
  default: "bg-surface text-ink-muted",
} as const;

function Tag({ children, tone }: Readonly<{ children: ReactNode; tone?: "active" | "selected" }>) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ${
        TAG_CLASSES[tone ?? "default"]
      }`}
    >
      {children}
    </span>
  );
}

function s3StatusLabel(status: ModelSummary["s3_status"]): string {
  if (status.error) return `S3: ${status.error}`;
  if (status.exists) return `S3: existe (VersionId ${status.version_id ?? "—"})`;
  return "S3: objeto inexistente";
}

function s3StatusClass(status: ModelSummary["s3_status"]): string {
  return status.exists ? "text-status-done" : "text-status-pending";
}

function activateLabel(active: boolean, activating: boolean): string {
  if (active) return "Ya es la activa";
  if (activating) return "Marcando…";
  return "Marcar como activa";
}

function VersionCard({
  model,
  selected,
  onSelect,
}: Readonly<{
  model: ModelSummary;
  selected: boolean;
  onSelect: () => void;
}>) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`flex w-full flex-col gap-2 rounded-2xl border bg-surface p-4 text-left transition-colors ${
        selected ? "border-accent-lilac" : "border-border hover:border-ink-muted"
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-semibold text-ink">{model.version}</span>
        {model.active && <Tag tone="active">activa</Tag>}
        {model.selected && <Tag tone="selected">candidato</Tag>}
        <span className="text-xs text-ink-muted">dataset {model.dataset_version}</span>
      </div>
      <p className="text-xs text-ink-muted">
        run {model.run_id.slice(0, 8)} · paquete {model.package_sha256.slice(0, 12)}…
      </p>
      <p className={`text-xs ${s3StatusClass(model.s3_status)}`}>
        {s3StatusLabel(model.s3_status)}
      </p>
    </button>
  );
}

function DetailPanel({
  detail,
  activating,
  actionError,
  onActivate,
}: Readonly<{
  detail: ModelDetail;
  activating: boolean;
  actionError: string;
  onActivate: () => void;
}>) {
  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-border bg-surface p-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="font-mono text-sm font-semibold text-ink">{detail.version}</h2>
        {detail.active && <Tag tone="active">activa</Tag>}
        <span className="text-xs text-ink-muted">dataset {detail.dataset_version}</span>
      </div>

      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
        <dt className="text-ink-muted">run_id</dt>
        <dd className="font-mono text-xs">{detail.run_id}</dd>
        <dt className="text-ink-muted">tipo de corrida</dt>
        <dd>{detail.run_kind ?? "—"}</dd>
        <dt className="text-ink-muted">manifiesto</dt>
        <dd className="font-mono text-xs">{detail.manifest_id ?? "—"}</dd>
        <dt className="text-ink-muted">VersionId registrado</dt>
        <dd className="font-mono text-xs">{detail.registered_version_id}</dd>
        <dt className="text-ink-muted">SHA-256 paquete</dt>
        <dd className="break-all font-mono text-xs">{detail.package_sha256}</dd>
        <dt className="text-ink-muted">S3</dt>
        <dd className="break-all font-mono text-xs">
          {detail.s3_bucket}/{detail.s3_key}
        </dd>
      </dl>

      {detail.card && (
        <div className="rounded-xl border border-border bg-surface p-3 text-sm whitespace-pre-wrap">
          {detail.card}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        {detail.download_url ? (
          <a
            href={detail.download_url}
            className="rounded-full border border-border px-4 py-2 text-sm font-medium"
          >
            Descargar paquete (URL prefirmada)
          </a>
        ) : (
          <span className="text-xs text-ink-muted">Sin URL de descarga (objeto ausente).</span>
        )}

        <button
          type="button"
          onClick={onActivate}
          disabled={activating || detail.active || !detail.s3_status.exists}
          className="rounded-full bg-ink px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {activateLabel(detail.active, activating)}
        </button>
      </div>

      {detail.s3_status.error && (
        <p className="text-sm text-status-pending">{detail.s3_status.error}</p>
      )}

      {actionError && <p className="text-sm text-status-pending">{actionError}</p>}
    </div>
  );
}

export function ModelsPage() {
  const [list, setList] = useState<ListState>({ status: "loading" });
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<DetailState>({ status: "idle" });
  const [activating, setActivating] = useState(false);
  const [actionError, setActionError] = useState("");

  const loadList = useCallback(() => {
    setList({ status: "loading" });
    getModels()
      .then((data) => setList({ status: "success", data }))
      .catch(() => setList({ status: "error" }));
  }, []);

  useEffect(loadList, [loadList]);

  async function openVersion(version: string) {
    setSelected(version);
    setActionError("");
    setDetail({ status: "loading" });
    try {
      const data = await getModel(version);
      setDetail({ status: "success", data });
    } catch (error) {
      setDetail({
        status: "error",
        message: error instanceof MlApiError ? error.message : "No se pudo cargar la versión.",
      });
    }
  }

  async function activate(version: string) {
    setActivating(true);
    setActionError("");
    try {
      const data = await setActiveModel(version);
      setDetail({ status: "success", data });
      loadList();
    } catch (error) {
      setActionError(
        error instanceof MlApiError ? error.message : "No se pudo marcar la versión activa."
      );
    } finally {
      setActivating(false);
    }
  }

  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <PageHeader
          title="Models"
          subtitle="Versiones publicadas en S3, con su estado en vivo, tarjeta y versión activa."
        />

        {list.status === "loading" && <p className="text-sm text-ink-muted">Cargando…</p>}
        {list.status === "error" && (
          <p className="text-sm text-ink-muted">No se pudo cargar el catálogo de modelos.</p>
        )}

        {list.status === "success" && list.data.status === "pending" && (
          <div className="rounded-2xl border border-border bg-surface p-4">
            <p className="font-medium text-ink">Sin versiones publicadas</p>
            <p className="mt-1 text-sm text-ink-muted">{list.data.message}</p>
          </div>
        )}

        {list.status === "success" && list.data.status === "ready" && (
          <>
            <div className="flex flex-col gap-2">
              <h2 className="text-sm font-medium text-ink">
                Versión activa para inferencia: {list.data.active_version ?? "ninguna"}
              </h2>
              {list.data.versions.length === 0 ? (
                <p className="text-sm text-ink-muted">El registro no tiene versiones.</p>
              ) : (
                <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {list.data.versions.map((model) => (
                    <li key={model.version}>
                      <VersionCard
                        model={model}
                        selected={selected === model.version}
                        onSelect={() => openVersion(model.version)}
                      />
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {detail.status === "loading" && (
              <p className="text-sm text-ink-muted">Cargando versión…</p>
            )}
            {detail.status === "error" && (
              <p className="text-sm text-ink-muted">{detail.message}</p>
            )}
            {detail.status === "success" && (
              <DetailPanel
                detail={detail.data}
                activating={activating}
                actionError={actionError}
                onActivate={() => activate(detail.data.version)}
              />
            )}
          </>
        )}
      </div>
    </main>
  );
}
