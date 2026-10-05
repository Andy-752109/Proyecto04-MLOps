import { type FormEvent, useEffect, useState } from "react";
import { PageHeader } from "@/pipeline/components/PageHeader";
import { StatusBadge } from "@/pipeline/components/StatusBadge";
import { useReleaseReport, useVersionsReport } from "@/pipeline/dataSource";
import type { DatasetRelease, QualityReport } from "@/pipeline/schemas";
import { MlApiError } from "../api/client";
import type { RunKind, TrainingConfig, TrainingJob } from "../api/contracts";
import { createTrainingJobRequestSchema } from "../api/contracts";
import {
  FROZEN_MIN_DELTA,
  FROZEN_PATIENCE,
  FROZEN_SEEDS,
  GRID,
  gridRowDefaults,
} from "../api/grid";
import { useManifestCounts, useManifestMeta } from "../api/provenance";
import { cancelTrainingJob, createTrainingJob, getTrainingJobs } from "../api/training";

// MLflow (docker-compose.yml) publica su UI en localhost:5050; "#/runs/<id>"
// es la ruta universal de MLflow que resuelve al experimento correcto sin
// que training_jobs tenga que guardar el experiment_id aparte.
const MLFLOW_RUN_URL = (runId: string) => `http://localhost:5050/#/runs/${runId}`;

// Secciones 4 y 6 de docs/decisiones-proyecto3.md: congelados para
// cualquier corrida -- el default ya no importa porque el campo ni se deja
// editar (ver NumberField con `disabled`), pero arranca en el valor real
// para que el formulario nunca muestre algo distinto a lo que de verdad se
// manda.
const DEFAULT_CONFIG: TrainingConfig = {
  ...GRID.r01,
  ...FROZEN_SEEDS,
  patience: FROZEN_PATIENCE,
  min_delta: FROZEN_MIN_DELTA,
};

const inputClass = "rounded border border-border bg-surface p-2";

type JobsState =
  | { status: "loading" }
  | { status: "error" }
  | { status: "success"; jobs: TrainingJob[] };

function SelectField<T extends string | number>({
  id,
  label,
  value,
  options,
  disabled,
  onChange,
}: {
  id: string;
  label: string;
  value: T;
  options: readonly T[];
  disabled?: boolean;
  onChange: (value: T) => void;
}) {
  return (
    <label htmlFor={id} className="flex flex-col gap-1 text-sm">
      {label}
      <select
        id={id}
        className={`${inputClass} disabled:opacity-60`}
        value={String(value)}
        disabled={disabled}
        onChange={(event) => {
          const raw = event.target.value;
          const match = options.find((option) => String(option) === raw);
          if (match !== undefined) onChange(match);
        }}
      >
        {options.map((option) => (
          <option key={String(option)} value={String(option)}>
            {String(option)}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumberField({
  id,
  label,
  value,
  step,
  disabled,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  step?: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label htmlFor={id} className="flex flex-col gap-1 text-sm">
      {label}
      <input
        id={id}
        className={`${inputClass} disabled:opacity-60`}
        type="number"
        step={step ?? "1"}
        value={Number.isFinite(value) ? value : ""}
        disabled={disabled}
        onChange={(event) =>
          onChange(event.target.value === "" ? Number.NaN : Number(event.target.value))
        }
      />
    </label>
  );
}

function ReleaseSelector({
  releases,
  selected,
  onSelect,
}: {
  releases: DatasetRelease[];
  selected: string | null;
  onSelect: (version: string) => void;
}) {
  return (
    <label htmlFor="release" className="flex flex-col gap-1 text-sm">
      Release
      <select
        id="release"
        className={inputClass}
        value={selected ?? ""}
        onChange={(event) => onSelect(event.target.value)}
      >
        <option value="" disabled>
          Elegir un release…
        </option>
        {releases.map((release) => (
          <option key={release.dataset_version} value={release.dataset_version}>
            {release.dataset_version}
          </option>
        ))}
      </select>
    </label>
  );
}

type QualityState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: QualityReport };

function Provenance({ release, quality }: { release: DatasetRelease; quality: QualityState }) {
  const meta = useManifestMeta(release.dataset_version);
  const counts = useManifestCounts(release.dataset_version);

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-border bg-surface p-4 text-sm">
      <div className="flex items-center gap-2">
        <span className="font-medium text-ink">Compuerta de calidad:</span>
        {quality.status === "loading" && <span className="text-ink-muted">cargando…</span>}
        {quality.status === "error" && <span className="text-ink-muted">{quality.message}</span>}
        {quality.status === "success" && <StatusBadge label={quality.data.status} />}
      </div>

      {meta.status === "loading" && <p className="text-ink-muted">Cargando procedencia…</p>}
      {meta.status === "error" && <p className="text-ink-muted">{meta.message}</p>}
      {meta.status === "success" && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          <dt className="text-ink-muted">manifest_id</dt>
          <dd className="font-mono text-xs">{meta.data.manifest_id}</dd>
          <dt className="text-ink-muted">SHA-256</dt>
          <dd className="break-all font-mono text-xs">{meta.data.manifest_sha256}</dd>
          <dt className="text-ink-muted">filas</dt>
          <dd>{meta.data.rows}</dd>
        </dl>
      )}

      {counts.status === "success" && (
        <div className="flex flex-col gap-1">
          <span className="font-medium text-ink">Conteos por clase y partición</span>
          <table className="text-left text-xs">
            <thead>
              <tr className="text-ink-muted">
                <th className="pr-3">partición</th>
                {Object.keys(counts.data.totals.classes).map((cls) => (
                  <th key={cls} className="pr-3">
                    {cls}
                  </th>
                ))}
                <th>desviación</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(counts.data.splits).map(([split, data]) => (
                <tr key={split}>
                  <td className="pr-3">{split}</td>
                  {Object.keys(counts.data.totals.classes).map((cls) => (
                    <td key={cls} className="pr-3">
                      {data.classes[cls]?.crops ?? 0}
                    </td>
                  ))}
                  <td>{data.deviation_pp.toFixed(2)} pp</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function LaunchForm({
  release,
  gateFailed,
  onLaunched,
}: {
  release: string;
  gateFailed: boolean;
  onLaunched: () => void;
}) {
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [runKind, setRunKind] = useState<RunKind>("smoke");
  const [gridRow, setGridRow] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  // Sección 7: una fila de la rejilla tiene un único combo válido -- al
  // elegirla en campaign, los 7 parámetros se autocompletan y se bloquean
  // (ver `gridLocked` más abajo) en vez de depender de que alguien los
  // copie bien a mano 10 veces.
  const gridDefaults = runKind === "campaign" ? gridRowDefaults(gridRow) : undefined;
  const gridLocked = gridDefaults !== undefined;
  useEffect(() => {
    if (gridDefaults) setConfig((current) => ({ ...current, ...gridDefaults }));
  }, [gridDefaults]);

  function field<K extends keyof TrainingConfig>(key: K) {
    return (value: TrainingConfig[K]) => setConfig((current) => ({ ...current, [key]: value }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");

    // Valida del lado del cliente antes de llamar a ml-api -- un valor
    // inválido (learning_rate fuera de rango, grid_row mal formado, etc.)
    // no debe generar ni un solo request, mismo criterio que QualityForm
    // (pipeline/pages/Settings.tsx).
    const parsed = createTrainingJobRequestSchema.safeParse({
      dataset_release: release,
      config,
      run_kind: runKind,
      grid_row: runKind === "campaign" ? gridRow || null : null,
    });
    if (!parsed.success) {
      setError(
        `Revisa los parámetros: ${parsed.error.issues.map((i) => i.path.join(".")).join(", ")}`
      );
      return;
    }

    setSubmitting(true);
    try {
      await createTrainingJob(parsed.data);
      onLaunched();
    } catch (err) {
      setError(err instanceof MlApiError ? err.message : "No se pudo lanzar el entrenamiento.");
    } finally {
      setSubmitting(false);
    }
  }

  const launchBlocked = gateFailed || submitting;

  return (
    <form onSubmit={submit} className="flex flex-col gap-4 rounded-2xl border border-border p-4">
      {gateFailed && (
        <p className="text-sm text-status-pending">
          Este release está <strong>failed</strong> en la compuerta de calidad: el lanzamiento queda
          bloqueado.
        </p>
      )}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <SelectField
          id="run_kind"
          label="Tipo de corrida"
          value={runKind}
          options={["smoke", "campaign"] as const}
          onChange={setRunKind}
        />
        {runKind === "campaign" && (
          <label htmlFor="grid_row" className="flex flex-col gap-1 text-sm">
            Fila de la rejilla (r01–r12)
            <input
              id="grid_row"
              className={inputClass}
              value={gridRow}
              placeholder="r01"
              onChange={(event) => setGridRow(event.target.value)}
            />
          </label>
        )}
        <SelectField
          id="optimizer"
          label="Optimizador"
          value={config.optimizer}
          options={["adam", "sgd"] as const}
          disabled={gridLocked}
          onChange={field("optimizer")}
        />
        <SelectField
          id="batch_size"
          label="Batch size"
          value={config.batch_size}
          options={[16, 32] as const}
          disabled={gridLocked}
          onChange={field("batch_size")}
        />
        <SelectField
          id="max_epochs"
          label="Épocas máximas"
          value={config.max_epochs}
          options={[15, 30] as const}
          disabled={gridLocked}
          onChange={field("max_epochs")}
        />
        <SelectField
          id="image_size"
          label="Tamaño de imagen"
          value={config.image_size}
          options={[128, 160] as const}
          disabled={gridLocked}
          onChange={field("image_size")}
        />
        <SelectField
          id="hidden_layers"
          label="Capas ocultas"
          value={config.hidden_layers}
          options={[0, 1] as const}
          disabled={gridLocked}
          onChange={field("hidden_layers")}
        />
        <SelectField
          id="dropout"
          label="Dropout"
          value={config.dropout}
          options={[0.0, 0.3, 0.5] as const}
          disabled={gridLocked}
          onChange={field("dropout")}
        />
        <NumberField
          id="learning_rate"
          label="Learning rate"
          step="0.0001"
          value={config.learning_rate}
          disabled={gridLocked}
          onChange={field("learning_rate")}
        />
        {/* Secciones 4 y 6: congeladas para cualquier corrida, nunca editables. */}
        <NumberField
          id="patience"
          label="Patience (congelado)"
          value={config.patience}
          disabled
          onChange={field("patience")}
        />
        <NumberField
          id="min_delta"
          label="Min delta (congelado)"
          step="0.001"
          value={config.min_delta}
          disabled
          onChange={field("min_delta")}
        />
        <NumberField
          id="seed_split"
          label="Semilla (split, congelada)"
          value={config.seed_split}
          disabled
          onChange={field("seed_split")}
        />
        <NumberField
          id="seed_train"
          label="Semilla (train, congelada)"
          value={config.seed_train}
          disabled
          onChange={field("seed_train")}
        />
        <NumberField
          id="seed_aug"
          label="Semilla (aug, congelada)"
          value={config.seed_aug}
          disabled
          onChange={field("seed_aug")}
        />
        <NumberField
          id="seed_model"
          label="Semilla (modelo, congelada)"
          value={config.seed_model}
          disabled
          onChange={field("seed_model")}
        />
      </div>

      {error && <p className="text-sm text-status-pending">{error}</p>}

      <button
        type="submit"
        disabled={launchBlocked}
        className="self-start rounded-full bg-ink px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
      >
        {submitting ? "Lanzando…" : "Lanzar entrenamiento"}
      </button>
    </form>
  );
}

// Revisión de Uriel sobre P3-09 ("no hay forma de cancelar"): solo
// queued/running son cancelables -- el resto ya es un estado terminal
// (mismo criterio que `TrainingJobNotCancellable` en ml_api/repository.py).
const CANCELLABLE_STATUSES: ReadonlySet<TrainingJob["status"]> = new Set(["queued", "running"]);

function JobRow({ job, onCancelled }: { job: TrainingJob; onCancelled: () => void }) {
  const [cancelling, setCancelling] = useState(false);

  async function handleCancel() {
    setCancelling(true);
    try {
      await cancelTrainingJob(job.id);
      onCancelled();
    } catch {
      // Lo más probable es que ya haya terminado entre que se pintó el botón
      // y el click (race normal) -- refrescar la lista alcanza, sin bloquear
      // el resto de la página por un solo cancel fallido.
      onCancelled();
    } finally {
      setCancelling(false);
    }
  }

  return (
    <li className="rounded-2xl border border-border bg-surface p-4 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-ink">{job.id}</span>
        <StatusBadge label={job.status} />
        <span className="text-ink-muted">
          {job.run_kind}
          {job.grid_row ? ` · ${job.grid_row}` : ""} · {job.dataset_release} ·{" "}
          {Math.round(job.progress * 100)}%
        </span>
        {job.mlflow_run_id && (
          <a
            href={MLFLOW_RUN_URL(job.mlflow_run_id)}
            target="_blank"
            rel="noreferrer"
            className="text-xs underline"
          >
            ver en MLflow
          </a>
        )}
        {CANCELLABLE_STATUSES.has(job.status) && (
          <button
            type="button"
            onClick={handleCancel}
            disabled={cancelling}
            className="ml-auto rounded-full border border-border px-3 py-1 text-xs disabled:opacity-40"
          >
            {cancelling ? "Cancelando…" : "Cancelar"}
          </button>
        )}
      </div>
      {job.error && <p className="mt-1 text-xs text-status-pending">{job.error}</p>}
    </li>
  );
}

/** Dueño de `useReleaseReport` para esta `release`: un solo fetch de
 * quality.json, compartido entre la procedencia (muestra el badge) y el
 * formulario (bloquea el lanzamiento si está failed) -- en vez de que cada
 * uno pida el mismo reporte por su cuenta. Solo se monta con un release ya
 * elegido, así que llamar el hook aquí adentro no viola las reglas de hooks
 * (la alternativa, llamarlo condicionalmente en TrainingPage, sí las viola). */
function ReleaseWorkspace({
  release,
  onLaunched,
}: {
  release: DatasetRelease;
  onLaunched: () => void;
}) {
  const quality = useReleaseReport("quality", release);
  return (
    <>
      <Provenance release={release} quality={quality} />
      <LaunchForm
        release={release.dataset_version}
        gateFailed={quality.status === "success" && quality.data.status === "failed"}
        onLaunched={onLaunched}
      />
    </>
  );
}

export function TrainingPage() {
  const versions = useVersionsReport();
  const [selectedRelease, setSelectedRelease] = useState<string | null>(null);
  const [jobsState, setJobsState] = useState<JobsState>({ status: "loading" });

  function loadJobs() {
    setJobsState({ status: "loading" });
    getTrainingJobs()
      .then((result) => setJobsState({ status: "success", jobs: result.jobs }))
      .catch(() => setJobsState({ status: "error" }));
  }

  useEffect(loadJobs, []);

  const release =
    versions.status === "success"
      ? (versions.data.releases.find((entry) => entry.dataset_version === selectedRelease) ?? null)
      : null;

  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <PageHeader
          title="Training"
          subtitle="Lanza entrenamientos reales desde el portal y sigue su progreso."
        />

        {versions.status === "loading" && <p className="text-sm text-ink-muted">Cargando…</p>}
        {versions.status === "error" && (
          <p className="text-sm text-ink-muted">No se pudo cargar el catálogo de releases.</p>
        )}
        {versions.status === "success" && (
          <ReleaseSelector
            releases={versions.data.releases}
            selected={selectedRelease}
            onSelect={setSelectedRelease}
          />
        )}

        {release && <ReleaseWorkspace release={release} onLaunched={loadJobs} />}

        <div className="flex flex-col gap-2">
          <h2 className="text-sm font-medium text-ink">Corridas</h2>
          {jobsState.status === "loading" && <p className="text-sm text-ink-muted">Cargando…</p>}
          {jobsState.status === "error" && (
            <p className="text-sm text-ink-muted">No se pudo cargar el estado de las corridas.</p>
          )}
          {jobsState.status === "success" && jobsState.jobs.length === 0 && (
            <p className="text-sm text-ink-muted">Sin corridas todavía.</p>
          )}
          {jobsState.status === "success" && jobsState.jobs.length > 0 && (
            <ul className="flex flex-col gap-2">
              {jobsState.jobs.map((job) => (
                <JobRow key={job.id} job={job} onCancelled={loadJobs} />
              ))}
            </ul>
          )}
        </div>
      </div>
    </main>
  );
}
