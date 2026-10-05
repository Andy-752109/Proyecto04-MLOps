import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { createInferenceSubmission, uploadImage } from "../src/lib/api/images";
import { InferencePage } from "../src/model/pages/Inference";

/**
 * SPEC-P3-16-INFERENCE: la página Inference (checklist del ticket #24) --
 * los dos flujos de POST /predict, las barras de probabilidad, la versión
 * usada, y "enviar a cola de anotación" (punto 5, solo en el flujo de
 * imagen nueva). `uploadImage`/`createInferenceSubmission` se mockean a
 * nivel de módulo -- `uploadImage` usa XMLHttpRequest, no `fetch`, así que
 * no lo cubre el mock de `serve()`.
 */

vi.mock("../src/lib/api/images", () => ({
  uploadImage: vi.fn(),
  createInferenceSubmission: vi.fn(),
}));

function response(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), { status });
}

const PREDICTION = {
  predicted_label: "cat",
  probabilities: { cat: 0.82, dog: 0.18 },
  model_version: "v1.0.0",
  checkpoint_sha256: "a".repeat(64),
};

function serve(predictResponse: unknown = PREDICTION, status = 200) {
  const fetcher = vi.fn((url: string, init?: RequestInit) => {
    if (url === "/ml-api/predict" && init?.method === "POST") {
      return Promise.resolve(response(predictResponse, status));
    }
    return Promise.resolve(response(null, 404));
  });
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.mocked(uploadImage).mockReset();
  vi.mocked(createInferenceSubmission).mockReset();
});

function fileInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector('input[type="file"]');
  if (!input) throw new Error("no se encontró el input de archivo");
  return input as HTMLInputElement;
}

it("subir una imagen válida llama a /predict y muestra la predicción con barras", async () => {
  const fetcher = serve();
  const { container } = render(<InferencePage />);

  const file = new File([new Uint8Array([1, 2, 3])], "cat.jpg", { type: "image/jpeg" });
  fireEvent.change(fileInput(container), { target: { files: [file] } });

  expect(await screen.findByText("v1.0.0")).toBeInTheDocument();
  expect((await screen.findAllByText("cat")).length).toBeGreaterThan(0);
  expect(screen.getByText("a".repeat(64))).toBeInTheDocument();
  expect(screen.getByText("82%")).toBeInTheDocument();
  expect(screen.getByText("18%")).toBeInTheDocument();

  const init = fetcher.mock.calls[0]?.[1];
  expect((init?.body as FormData).get("image")).toBe(file);
});

it("un archivo con tipo no soportado no llama a la API", async () => {
  const fetcher = serve();
  const { container } = render(<InferencePage />);

  const file = new File(["no soy una imagen"], "nota.txt", { type: "text/plain" });
  fireEvent.change(fileInput(container), { target: { files: [file] } });

  expect(await screen.findByText(/Tipo de archivo no soportado/)).toBeInTheDocument();
  expect(fetcher).not.toHaveBeenCalled();
});

it("un error del servidor se muestra en vez de la predicción", async () => {
  serve({ error: "tipo de archivo no soportado" }, 400);
  const { container } = render(<InferencePage />);

  const file = new File([new Uint8Array([1, 2, 3])], "cat.jpg", { type: "image/jpeg" });
  fireEvent.change(fileInput(container), { target: { files: [file] } });

  expect(await screen.findByText("tipo de archivo no soportado")).toBeInTheDocument();
});

it("el flujo de recorte existente manda image_id y annotation_id, no un archivo", async () => {
  const fetcher = serve();
  render(<InferencePage />);

  fireEvent.click(screen.getByRole("tab", { name: "Recorte existente" }));
  fireEvent.change(screen.getByLabelText("image_id"), { target: { value: "7" } });
  fireEvent.change(screen.getByLabelText("annotation_id"), { target: { value: "42" } });
  fireEvent.click(screen.getByRole("button", { name: "Predecir" }));

  expect(await screen.findByText("v1.0.0")).toBeInTheDocument();
  const init = fetcher.mock.calls[0]?.[1];
  const body = init?.body as FormData;
  expect(body.get("image_id")).toBe("7");
  expect(body.get("annotation_id")).toBe("42");
  expect(body.get("image")).toBeNull();
});

it("un image_id/annotation_id no numérico no llama a la API", async () => {
  const fetcher = serve();
  render(<InferencePage />);

  fireEvent.click(screen.getByRole("tab", { name: "Recorte existente" }));
  fireEvent.change(screen.getByLabelText("image_id"), { target: { value: "abc" } });
  fireEvent.change(screen.getByLabelText("annotation_id"), { target: { value: "42" } });
  fireEvent.click(screen.getByRole("button", { name: "Predecir" }));

  expect(await screen.findByText(/deben ser enteros/)).toBeInTheDocument();
  expect(fetcher).not.toHaveBeenCalled();
});

// --- enviar a cola de anotación (checklist punto 5) -----------------------------

it("enviar a cola de anotación sube la imagen y guarda la sugerencia del modelo", async () => {
  serve();
  vi.mocked(uploadImage).mockResolvedValue({
    id: 42,
    filename: "cat.jpg",
    storageKey: "images/abc",
    width: 64,
    height: 64,
  });
  vi.mocked(createInferenceSubmission).mockResolvedValue({ id: 7 });

  const { container } = render(<InferencePage />);
  const file = new File([new Uint8Array([1, 2, 3])], "cat.jpg", { type: "image/jpeg" });
  fireEvent.change(fileInput(container), { target: { files: [file] } });

  const queueButton = await screen.findByRole("button", { name: "Enviar a cola de anotación" });
  fireEvent.click(queueButton);

  expect(await screen.findByText(/Enviada a la cola \(imagen #42\)/)).toBeInTheDocument();
  expect(uploadImage).toHaveBeenCalledWith(file);
  expect(createInferenceSubmission).toHaveBeenCalledWith(42, {
    predictedLabel: "cat",
    probabilities: PREDICTION.probabilities,
    modelVersion: "v1.0.0",
    checkpointSha256: "a".repeat(64),
  });
});

it("un error al enviar a cola se muestra sin perder la predicción", async () => {
  serve();
  vi.mocked(uploadImage).mockRejectedValue(new Error("falló"));

  const { container } = render(<InferencePage />);
  const file = new File([new Uint8Array([1, 2, 3])], "cat.jpg", { type: "image/jpeg" });
  fireEvent.change(fileInput(container), { target: { files: [file] } });

  fireEvent.click(await screen.findByRole("button", { name: "Enviar a cola de anotación" }));

  expect(await screen.findByText("No se pudo enviar a la cola.")).toBeInTheDocument();
  expect((await screen.findAllByText("cat")).length).toBeGreaterThan(0);
});

it("el flujo de recorte existente no ofrece enviar a cola", async () => {
  serve();
  render(<InferencePage />);

  fireEvent.click(screen.getByRole("tab", { name: "Recorte existente" }));
  fireEvent.change(screen.getByLabelText("image_id"), { target: { value: "7" } });
  fireEvent.change(screen.getByLabelText("annotation_id"), { target: { value: "42" } });
  fireEvent.click(screen.getByRole("button", { name: "Predecir" }));

  expect(await screen.findByText("v1.0.0")).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Enviar a cola de anotación" })
  ).not.toBeInTheDocument();
});
