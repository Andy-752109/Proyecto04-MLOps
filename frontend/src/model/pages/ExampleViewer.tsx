import { useState } from "react";
import type { EvaluationExample } from "../api/contracts";
import { cropImageUrl } from "../api/evaluation";
import { percent } from "./format";

const KIND_SUCCESS = "Acierto";
const KIND_ERROR = "Error";

type ViewerItem = Readonly<{
  kind: typeof KIND_SUCCESS | typeof KIND_ERROR;
  example: EvaluationExample;
}>;

/** Aciertos primero, luego errores, cada ejemplo con su etiqueta. */
function buildItems(
  successes: readonly EvaluationExample[],
  errors: readonly EvaluationExample[]
): ViewerItem[] {
  return [
    ...successes.map((example) => ({ kind: KIND_SUCCESS, example }) as const),
    ...errors.map((example) => ({ kind: KIND_ERROR, example }) as const),
  ];
}

/** Visor con Anterior/Siguiente sobre los aciertos y errores del test (P3-19). */
export function ExampleViewer({
  successes,
  errors,
}: Readonly<{
  successes: readonly EvaluationExample[];
  errors: readonly EvaluationExample[];
}>) {
  const items = buildItems(successes, errors);
  const [index, setIndex] = useState(0);
  const last = items.length - 1;
  // Si el reporte cambia y trae menos ejemplos, muestra el último en vez de desaparecer.
  const position = Math.min(index, last);
  const current = items[position];
  if (!current) return null;
  const { kind, example } = current;

  return (
    <section
      aria-label="Explorador de ejemplos"
      className="flex flex-col gap-3 rounded-2xl border border-border bg-surface p-4"
    >
      <h2 className="text-sm font-medium text-ink">Explorador de ejemplos</h2>
      <div className="flex items-center justify-between text-xs text-ink-muted">
        <span className="font-medium text-ink">{kind}</span>
        <span>{`${position + 1} de ${items.length}`}</span>
      </div>
      <img
        src={cropImageUrl(example.crop_id)}
        alt={`Recorte seleccionado ${example.crop_id}`}
        className="aspect-square w-full max-w-xs self-center rounded-lg object-cover"
      />
      <p className="font-mono text-[11px] text-ink-muted">{example.crop_id}</p>
      <p className="text-sm text-ink">
        real <strong>{example.true_class}</strong>
      </p>
      <p className="text-sm text-ink">
        pred <strong>{example.predicted_class}</strong> · {percent(example.probability)}
      </p>
      <div className="flex gap-2">
        <button
          type="button"
          disabled={position === 0}
          onClick={() => setIndex(Math.max(0, position - 1))}
          className="rounded-full border border-border px-4 py-1.5 text-sm text-ink disabled:opacity-40"
        >
          Anterior
        </button>
        <button
          type="button"
          disabled={position === last}
          onClick={() => setIndex(Math.min(last, position + 1))}
          className="rounded-full bg-ink px-4 py-1.5 text-sm text-white disabled:opacity-40"
        >
          Siguiente
        </button>
      </div>
    </section>
  );
}
