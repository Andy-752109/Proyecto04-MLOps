import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MLflowRun } from "../../lib/experimentsUtils";
import { ExperimentsPage } from "./Experiments";

/**
 * P3-17 (#27, checklist punto 3): auditoría de las 5 páginas -- a esta le
 * faltaba una prueba propia (solo tenía `experimentsUtils.test.ts`, que
 * prueba las funciones puras, y una verificación superficial en
 * `tests/model-nav.test.tsx`). Se mockea `fetch` (nunca se toca ml-api real).
 */

function aRun(runId: string, bestValLoss: number, estado?: string): MLflowRun {
  return {
    info: { run_id: runId, experiment_id: "0" },
    data: {
      metrics: { best_val_loss: bestValLoss },
      tags: estado ? { estado } : {},
    },
  };
}

function mockRuns(runs: MLflowRun[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      if (url === "/ml-api/experiments/1/runs") {
        return Promise.resolve(
          new Response(JSON.stringify(runs), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }
      return Promise.resolve(new Response(null, { status: 404 }));
    })
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("ExperimentsPage", () => {
  it("muestra las corridas reales ordenadas por best_val_loss ascendente", async () => {
    mockRuns([aRun("run-high-loss-aaaaaaaa", 0.9), aRun("run-low-loss-bbbbbbbbb", 0.1)]);
    render(<ExperimentsPage />);

    await waitFor(() => {
      expect(screen.getAllByRole("row")).toHaveLength(3); // encabezado + 2 corridas
    });

    const rows = screen.getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("run-low-"); // menor loss primero (orden asc por default)
    expect(rows[1]).toHaveTextContent("run-high");
  });

  it("'Solo Válidas' filtra las corridas sin tag estado=valida", async () => {
    mockRuns([aRun("run-valida-aaaaaaaaaa", 0.1, "valida"), aRun("run-mala-bbbbbbbbbbb", 0.2)]);
    render(<ExperimentsPage />);

    await waitFor(() => {
      expect(screen.getAllByRole("row")).toHaveLength(3);
    });

    fireEvent.click(screen.getByRole("button", { name: "Solo Válidas" }));

    await waitFor(() => {
      expect(screen.getAllByRole("row")).toHaveLength(2); // encabezado + 1 válida
    });
    expect(screen.getByText(/run-vali/)).toBeInTheDocument();
  });

  it("cambiar el orden invierte las filas", async () => {
    mockRuns([aRun("run-aaa-11111111111", 0.1), aRun("run-bbb-22222222222", 0.5)]);
    render(<ExperimentsPage />);
    await waitFor(() => expect(screen.getAllByRole("row")).toHaveLength(3));

    fireEvent.click(screen.getByRole("button", { name: /Orden Loss/ }));

    const rows = screen.getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("run-bbb-");
    expect(rows[1]).toHaveTextContent("run-aaa-");
  });

  it("si ml-api no responde, no truena -- queda sin corridas", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(null, { status: 500 })))
    );
    render(<ExperimentsPage />);

    await waitFor(() => {
      expect(screen.getAllByRole("row")).toHaveLength(1); // solo el encabezado
    });
  });

  it("cada fila enlaza a la UI nativa de MLflow para esa corrida", async () => {
    mockRuns([aRun("run-link-aaaaaaaaaaa", 0.1)]);
    render(<ExperimentsPage />);

    const link = await screen.findByRole("link", { name: "Ver gráficas nativas" });
    expect(link).toHaveAttribute("href", expect.stringContaining("run-link-aaaaaaaaaaa"));
  });
});
