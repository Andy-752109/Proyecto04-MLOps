import { cleanup, render } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { EvaluationPage } from "@/model/pages/Evaluation";

/**
 * Utilidades compartidas por las pruebas de la página Evaluation (P3-15/P3-19).
 * Los datos son fixtures: la pantalla no tiene ninguna cifra escrita a mano.
 */

/** Un ejemplo de test; `source_image_id` sale del prefijo de `crop_id` ("000275_…" -> "275"). */
export function example(
  cropId: string,
  trueClass: string,
  predictedClass: string,
  probability: number
) {
  return {
    crop_id: cropId,
    source_image_id: String(Number(cropId.split("_")[0])),
    true_class: trueClass,
    predicted_class: predictedClass,
    probability,
  };
}

/** Respuesta `ready` de `GET /ml-api/evaluation`, con las cifras de reports/evaluation/metrics.json. */
export const READY = {
  status: "ready",
  run_id: "7e7b4a4b35464cfebb6b41714a3ad931",
  release: "v0.1.1",
  manifest_id: "v0.1.1-53fc84fdaa07",
  manifest_sha256: "5".repeat(64),
  checkpoint_sha256: "f".repeat(64),
  selected_at: "2026-10-02T05:43:20.705345Z",
  grid_row: "r02",
  best_val_accuracy: 0.9618320610687023,
  best_val_macro_f1: 0.9618231625575566,
  best_val_loss: 0.12332657724618912,
  classes: ["cat", "dog"],
  accuracy: 0.9863013698630136,
  macro_f1: 0.9862081995087851,
  confusion_matrix: [
    [39, 0],
    [1, 33],
  ],
  per_class: {
    cat: { precision: 0.975, recall: 1.0, f1: 0.9873417721518987, support: 39 },
    dog: { precision: 1.0, recall: 0.9705882352941176, f1: 0.9850746268656716, support: 34 },
  },
  total: 73,
  baseline_majority_accuracy: 0.5342465753424658,
  most_confused_class: "dog",
  recall_per_class: { cat: 1.0, dog: 0.9705882352941176 },
  accuracy_hides_low_recall: false,
  successes: [example("000008_000001", "cat", "cat", 0.9534333348274231)],
  errors: [example("000275_000320", "dog", "cat", 0.5771472454071045)],
};

/** Sustituye `fetch` por una respuesta JSON fija (nunca se toca ml-api real). */
export function mockEvaluation(payload: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    )
  );
}

/** `mockEvaluation` + render de la página. */
export function renderEvaluation(payload: unknown) {
  mockEvaluation(payload);
  return render(<EvaluationPage />);
}

/** Coincide con el `<p>` cuyo texto completo (incluido el `<strong>`) es `text`. */
export function paragraph(text: string) {
  return (_content: string, element: Element | null) =>
    element?.tagName === "P" && element.textContent === text;
}

/** Registra la limpieza de DOM y de `fetch` falso tras cada prueba del archivo que la llama. */
export function setupEvaluationTests() {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });
}
