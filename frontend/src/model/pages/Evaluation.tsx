import { useEffect, useState } from "react";
import { PageHeader } from "@/pipeline/components/PageHeader";
import type { EvaluationExample, EvaluationReport } from "../api/contracts";
import { cropImageUrl, getEvaluation, PREDICTIONS_CSV_URL } from "../api/evaluation";
import { ExampleViewer } from "./ExampleViewer";
import { percent } from "./format";

type State =
  | { status: "loading" }
  | { status: "error" }
  | { status: "success"; data: Awaited<ReturnType<typeof getEvaluation>> };

function MetricCard({
  label,
  value,
  hint,
}: Readonly<{ label: string; value: string; hint?: string }>) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-4">
      <p className="text-xs text-ink-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-ink">{value}</p>
      {hint && <p className="mt-1 text-xs text-ink-muted">{hint}</p>}
    </div>
  );
}

function Provenance({ report }: Readonly<{ report: EvaluationReport }>) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-2xl border border-border bg-surface p-4 text-sm">
      <dt className="text-ink-muted">Candidato</dt>
      <dd className="font-medium text-ink">
        {report.grid_row} · {report.run_id.slice(0, 8)}
      </dd>
      <dt className="text-ink-muted">Release</dt>
      <dd>{report.release}</dd>
      <dt className="text-ink-muted">Manifiesto</dt>
      <dd className="font-mono text-xs">{report.manifest_id}</dd>
      <dt className="text-ink-muted">SHA-256 manifiesto</dt>
      <dd className="break-all font-mono text-xs">{report.manifest_sha256}</dd>
      <dt className="text-ink-muted">SHA-256 checkpoint</dt>
      <dd className="break-all font-mono text-xs">{report.checkpoint_sha256}</dd>
      <dt className="text-ink-muted">Seleccionado</dt>
      <dd>{report.selected_at}</dd>
    </dl>
  );
}

function ConfusionMatrix({ report }: Readonly<{ report: EvaluationReport }>) {
  return (
    <div className="flex flex-col gap-2 overflow-x-auto">
      <h2 className="text-sm font-medium text-ink">
        Matriz de confusión (filas = real, columnas = predicha)
      </h2>
      <table className="w-fit text-left text-sm">
        <thead>
          <tr className="text-ink-muted">
            <th className="pr-3" />
            {report.classes.map((cls) => (
              <th key={cls} className="px-3">
                pred. {cls}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {report.confusion_matrix.map((row, rowIndex) => (
            <tr key={report.classes[rowIndex]}>
              <th className="pr-3 text-right font-normal text-ink-muted">
                real {report.classes[rowIndex]}
              </th>
              {row.map((value, columnIndex) => (
                <td
                  key={report.classes[columnIndex]}
                  className={`px-3 text-center ${
                    rowIndex === columnIndex ? "font-semibold text-status-done" : ""
                  }`}
                >
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Gallery({ title, examples }: Readonly<{ title: string; examples: EvaluationExample[] }>) {
  if (examples.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <h2 className="text-sm font-medium text-ink">{title}</h2>
        <p className="text-sm text-ink-muted">Sin ejemplos en el test.</p>
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <h2 className="text-sm font-medium text-ink">
        {title} ({examples.length})
      </h2>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {examples.map((example) => (
          <li
            key={example.crop_id}
            className="flex flex-col gap-1 rounded-2xl border border-border bg-surface p-2"
          >
            <img
              src={cropImageUrl(example.crop_id)}
              alt={`Recorte ${example.crop_id}`}
              loading="lazy"
              className="aspect-square w-full rounded-lg object-cover"
            />
            <p className="font-mono text-[11px] text-ink-muted">{example.crop_id}</p>
            <p className="text-xs text-ink">
              real <strong>{example.true_class}</strong>
            </p>
            <p className="text-xs text-ink">
              pred <strong>{example.predicted_class}</strong> · {percent(example.probability)}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ReportView({ report }: Readonly<{ report: EvaluationReport }>) {
  return (
    <>
      <Provenance report={report} />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard
          label="Accuracy (test)"
          value={percent(report.accuracy)}
          hint={`${Math.round(report.accuracy * report.total)} / ${report.total} recortes`}
        />
        <MetricCard label="F1 macro" value={report.macro_f1.toFixed(4)} />
        <MetricCard
          label="Baseline clase mayoritaria"
          value={percent(report.baseline_majority_accuracy)}
        />
        <MetricCard
          label="Accuracy (validación)"
          value={percent(report.best_val_accuracy)}
          hint={`r02 · F1 ${report.best_val_macro_f1.toFixed(4)}`}
        />
      </div>

      <div className="flex flex-col gap-2 overflow-x-auto">
        <h2 className="text-sm font-medium text-ink">Métricas por clase</h2>
        <table className="w-fit text-left text-sm">
          <thead>
            <tr className="text-ink-muted">
              <th className="pr-4">clase</th>
              <th className="pr-4">precisión</th>
              <th className="pr-4">recall</th>
              <th className="pr-4">F1</th>
              <th>support</th>
            </tr>
          </thead>
          <tbody>
            {report.classes.map((cls) => {
              const metrics = report.per_class[cls];
              if (!metrics) return null;
              return (
                <tr key={cls}>
                  <td className="pr-4 font-medium">{cls}</td>
                  <td className="pr-4">{metrics.precision.toFixed(4)}</td>
                  <td className="pr-4">{metrics.recall.toFixed(4)}</td>
                  <td className="pr-4">{metrics.f1.toFixed(4)}</td>
                  <td>{metrics.support}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <ConfusionMatrix report={report} />

      <div className="rounded-2xl border border-border bg-surface p-4 text-sm">
        <p>
          Clase más confundida: <strong>{report.most_confused_class}</strong>.
        </p>
        <p className="mt-1 text-ink-muted">
          {report.accuracy_hides_low_recall
            ? "La accuracy oculta un recall bajo en alguna clase."
            : "Ninguna clase tiene un recall bajo: la accuracy no esconde fallas."}
        </p>
      </div>

      <ExampleViewer successes={report.successes} errors={report.errors} />

      <Gallery title="Aciertos" examples={report.successes} />
      <Gallery title="Errores" examples={report.errors} />

      <a
        href={PREDICTIONS_CSV_URL}
        download
        className="self-start rounded-full bg-ink px-4 py-2 text-sm font-medium text-white"
      >
        Descargar predictions.csv
      </a>
    </>
  );
}

export function EvaluationPage() {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    getEvaluation(controller.signal)
      .then((data) => setState({ status: "success", data }))
      .catch(() => setState({ status: "error" }));
    return () => controller.abort();
  }, []);

  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <PageHeader
          title="Evaluation"
          subtitle="Evaluación final del candidato sobre el 10% de prueba, medida una sola vez."
        />

        {state.status === "loading" && <p className="text-sm text-ink-muted">Cargando…</p>}
        {state.status === "error" && (
          <p className="text-sm text-ink-muted">No se pudo cargar la evaluación.</p>
        )}

        {state.status === "success" && state.data.status === "selection_not_closed" && (
          <div className="rounded-2xl border border-border bg-surface p-4">
            <p className="font-medium text-ink">Selección no cerrada</p>
            <p className="mt-1 text-sm text-ink-muted">{state.data.message}</p>
          </div>
        )}

        {state.status === "success" && state.data.status === "pending" && (
          <p className="text-sm text-ink-muted">
            La selección está cerrada, pero la evaluación final todavía no está disponible:{" "}
            {state.data.message}
          </p>
        )}

        {state.status === "success" && state.data.status === "ready" && (
          <ReportView report={state.data} />
        )}
      </div>
    </main>
  );
}
