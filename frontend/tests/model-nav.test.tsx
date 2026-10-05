import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";

/**
 * SPEC-P3-03-NAV: las 5 rutas de "Modelo" existen y se alcanzan desde el
 * menú (issue #6, checklist punto 9). No monta ninguna de las páginas de
 * anotación/pipeline que ya prueba pipeline-ui.test.tsx.
 */

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  );
}

function mockMlApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      if (url === "/ml-api/training/jobs") return jsonResponse({ jobs: [] });
      if (url === "/ml-api/experiments") {
        return jsonResponse({ status: "pending", ticket: "P3-12", message: "todavía no" });
      }
      if (url === "/ml-api/evaluation") {
        return jsonResponse({ status: "pending", ticket: "P3-13", message: "todavía no" });
      }
      if (url === "/ml-api/models") {
        return jsonResponse({ status: "pending", ticket: "P3-14", message: "todavía no" });
      }
      if (url === "/ml-api/inference") {
        return jsonResponse({ status: "pending", ticket: "P3-16", message: "todavía no" });
      }
      return jsonResponse(null, 404);
    })
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function renderAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

describe("SPEC-P3-03-NAV - las 5 rutas de Modelo existen y se alcanzan desde el menú", () => {
  it("Training (/model/training) muestra 'Sin corridas todavía' con la lista vacía real", async () => {
    mockMlApi();
    renderAt("/model/training");

    expect(await screen.findByRole("heading", { name: "Training" })).toBeInTheDocument();
    expect(await screen.findByText("Sin corridas todavía.")).toBeInTheDocument();
  });

  it("navega a Experiments, Evaluation, Models e Inference desde el nav", async () => {
    mockMlApi();
    renderAt("/model/training");
    await screen.findByRole("heading", { name: "Training" });

    fireEvent.click(screen.getByRole("link", { name: "Experiments" }));
    expect(await screen.findByRole("heading", { name: /Experimentos de MLflow en vivo/i })).toBeInTheDocument();
    expect(await screen.findByText(/Experimentos de MLflow en vivo/i)).toBeInTheDocument();

    // P3-15: Evaluation ya no dice "Pendiente"; sin evaluación final muestra
    // el motivo real de ml-api.
    fireEvent.click(screen.getByRole("link", { name: "Evaluation" }));
    expect(await screen.findByRole("heading", { name: "Evaluation" })).toBeInTheDocument();
    expect(await screen.findByText(/todavía no/)).toBeInTheDocument();

    // P3-15: Models distingue "sin versiones publicadas" del pendiente genérico.
    fireEvent.click(screen.getByRole("link", { name: "Models" }));
    expect(await screen.findByRole("heading", { name: "Models" })).toBeInTheDocument();
    expect(await screen.findByText("Sin versiones publicadas")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("link", { name: "Inference" }));
    expect(await screen.findByRole("heading", { name: "Inference" })).toBeInTheDocument();
    expect(await screen.findByRole("tab", { name: "Imagen nueva" })).toBeInTheDocument();
  });

  it("las páginas pendientes y Experiments renderizan correctamente aunque ml-api no responda", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => jsonResponse(null, 404))
    );
    renderAt("/experiments");

    expect(await screen.findByRole("heading", { name: /Experimentos de MLflow en vivo/i })).toBeInTheDocument();
    expect(await screen.findByText(/Experimentos de MLflow en vivo/i)).toBeInTheDocument();
  });

  it("Training se conecta al menú de Training desde Experiments", async () => {
    mockMlApi();
    renderAt("/experiments");
    await screen.findByRole("heading", { name: /Experimentos de MLflow en vivo/i });

    fireEvent.click(screen.getByRole("link", { name: "Training" }));

    expect(await screen.findByRole("heading", { name: "Training" })).toBeInTheDocument();
  });
});
