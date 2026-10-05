import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EvaluationPage } from "./Evaluation";

/**
 * P3-15: la página refleja el candado de P3-11 y, cuando la selección está
 * cerrada y existen los reportes de P3-13, muestra las cifras reales y la
 * galería de recortes. Se mockea `fetch` (nunca se toca ml-api real).
 */

function mockEvaluation(payload: unknown) {
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

const READY = {
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
  successes: [
    {
      crop_id: "000008_000001",
      source_image_id: "8",
      true_class: "cat",
      predicted_class: "cat",
      probability: 0.9534333348274231,
    },
  ],
  errors: [
    {
      crop_id: "000275_000320",
      source_image_id: "275",
      true_class: "dog",
      predicted_class: "cat",
      probability: 0.5771472454071045,
    },
  ],
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("EvaluationPage (P3-15)", () => {
  it("muestra el candado de P3-11 cuando la selección no está cerrada", async () => {
    mockEvaluation({
      status: "selection_not_closed",
      ticket: "P3-11",
      message: "selección no cerrada (no existe reports/selection.json)",
    });

    render(<EvaluationPage />);

    expect(await screen.findByText("Selección no cerrada")).toBeInTheDocument();
    expect(
      screen.getByText("selección no cerrada (no existe reports/selection.json)")
    ).toBeInTheDocument();
  });

  it("pinta las cifras reales y la galería con el recorte servido por ml-api", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);

    expect(await screen.findByText("98.63%")).toBeInTheDocument();
    expect(screen.getByText("72 / 73 recortes")).toBeInTheDocument();

    const crop = await screen.findByAltText("Recorte 000275_000320");
    expect(crop).toHaveAttribute("src", "/ml-api/crops/000275_000320");

    expect(screen.getByRole("link", { name: "Descargar predictions.csv" })).toHaveAttribute(
      "href",
      "/reports/evaluation/predictions.csv"
    );

    await waitFor(() =>
      expect(fetch).toHaveBeenCalledWith("/ml-api/evaluation", expect.anything())
    );
  });
});
