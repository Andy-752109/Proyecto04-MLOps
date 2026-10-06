import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { edgeCaptureListSchema } from "../src/edge/api/contracts";

// Copias locales de contracts/examples/valid-cat-no-crop.json y
// valid-dog-with-crop.json. El build Docker solo incluye ./frontend.
const catEvent = {
  schema_version: "1",
  capture_id: "8f9f60e4-6d7a-4d92-8cc0-45e0b192cd65",
  captured_at: "2026-10-05T12:34:56-06:00",
  device_id: "edge-demo-01",
  model_version: "optimized-v1",
  model_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  runtime: "local-runtime",
  predicted_class: "cat",
  confidence: 0.93,
  probabilities: { cat: 0.93, dog: 0.07 },
  crop: null,
  preprocess_ms: 12.5,
  inference_ms: 38.2,
  image_key: "edge-captures/v1/images/8f9f60e4-6d7a-4d92-8cc0-45e0b192cd65.jpg",
};
const dogEvent = {
  schema_version: "1",
  capture_id: "d16b8f30-a4c7-4eca-b32f-f8878a617ac4",
  captured_at: "2026-10-05T18:35:10Z",
  device_id: "edge-demo-01",
  model_version: "optimized-v1",
  model_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  runtime: "local-runtime",
  predicted_class: "dog",
  confidence: 0.82,
  probabilities: { cat: 0.18, dog: 0.82 },
  crop: { x: 12, y: 8, width: 80, height: 60, frame_width: 320, frame_height: 240 },
  preprocess_ms: 12.5,
  inference_ms: 38.2,
  image_key: "edge-captures/v1/images/d16b8f30-a4c7-4eca-b32f-f8878a617ac4.jpg",
};

// Direcciones ficticias: ninguna prueba utiliza URLs prefirmadas ni red real.
const cat = {
  ...catEvent,
  received_at: "2026-10-06T17:17:04Z",
  image_url: "https://example.invalid/cat.jpg",
};
const dog = {
  ...dogEvent,
  received_at: "2026-10-06T17:18:04Z",
  image_url: "https://example.invalid/dog.jpg",
};
const bucket = "test-edge-captures";

function response(items: unknown[], total = items.length) {
  return new Response(JSON.stringify({ items, total, bucket }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function renderAt(path = "/edge/captures") {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

function listItems() {
  return within(screen.getByRole("list", { name: "Capturas Edge" })).getAllByRole("listitem");
}

function deferredResponse() {
  let resolve: (value: Response) => void = () => {
    throw new Error("La petición todavía no inició");
  };
  const promise = new Promise<Response>((complete) => {
    resolve = complete;
  });
  return { promise, resolve: (value: Response) => resolve(value) };
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("P4-08 Capturas Edge", () => {
  it("muestra skeleton mientras carga y pide limit=100 sin caché con AbortSignal", async () => {
    const fetcher = vi.fn(() => new Promise<Response>(() => {}));
    vi.stubGlobal("fetch", fetcher);
    renderAt();

    expect(screen.getByRole("status", { name: "Cargando capturas edge" })).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Capturas Edge" })).not.toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledWith(
      "/ml-api/edge/captures?limit=100",
      expect.objectContaining({ cache: "no-store", signal: expect.any(AbortSignal) })
    );
  });

  it("muestra el estado vacío sin cambiar el texto de otros EmptyState", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response([])));
    renderAt();

    expect(await screen.findByText("No hay capturas edge")).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Capturas Edge" })).not.toBeInTheDocument();
  });

  it("muestra error y permite reintentar una respuesta fallida", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(new Response('{"error":"detalle interno no visible"}', { status: 503 }))
      .mockResolvedValueOnce(response([cat]));
    vi.stubGlobal("fetch", fetcher);
    renderAt();

    expect(await screen.findByText("No se pudieron cargar las capturas edge.")).toBeInTheDocument();
    expect(screen.getByText("El servicio de capturas o AWS no está disponible temporalmente. Inténtalo de nuevo.")).toBeInTheDocument();
    expect(screen.queryByText("detalle interno no visible")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reintentar" }));
    expect(await screen.findByText(cat.capture_id)).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("distingue fallo de conexión y respuesta incompatible sin mostrar detalles internos", async () => {
    const fetcher = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("detalle privado de red"))
      .mockResolvedValueOnce(response([{ ...cat, schema_version: "2" }]));
    vi.stubGlobal("fetch", fetcher);
    renderAt();

    expect(await screen.findByText("No se pudo conectar con ml-api. Comprueba la conexión e inténtalo de nuevo.")).toBeInTheDocument();
    expect(screen.queryByText("detalle privado de red")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reintentar" }));
    expect(await screen.findByText("ml-api devolvió una respuesta inválida o incompatible con el contrato de capturas.")).toBeInTheDocument();
    expect(screen.queryByText("schema_version")).not.toBeInTheDocument();
  });

  it("muestra imagen, metadatos, fechas con offset y el crop sobre el frame", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response([cat, dog])));
    renderAt();

    expect(await screen.findByText(cat.capture_id)).toBeInTheDocument();
    const [dogCard, catCard] = listItems();
    if (!dogCard || !catCard) throw new Error("Faltan tarjetas de captura");
    expect(dogCard).toHaveTextContent(dog.capture_id);
    expect(catCard).toHaveTextContent(cat.capture_id);
    expect(catCard).toHaveTextContent("cat");
    expect(catCard).toHaveTextContent(/93\s*%/);
    expect(catCard).toHaveTextContent("5 oct 2026, 12:34:56 (UTC-06:00)");
    expect(catCard).toHaveTextContent("6 oct 2026, 17:17:04 (UTC+00:00)");
    expect(within(catCard).getByText("5 oct 2026, 12:34:56 (UTC-06:00)")).toHaveAttribute(
      "datetime",
      cat.captured_at
    );
    expect(within(catCard).getByText("6 oct 2026, 17:17:04 (UTC+00:00)")).toHaveAttribute(
      "datetime",
      cat.received_at
    );
    expect(catCard).toHaveTextContent(cat.device_id);
    expect(catCard).toHaveTextContent(cat.model_version);
    expect(catCard).toHaveTextContent("Sin recorte");
    expect(within(catCard).getByRole("img")).toHaveAttribute("src", cat.image_url);
    expect(catCard.querySelector("svg")).toBeNull();
    expect(dogCard).toHaveTextContent("x=12, y=8, ancho=80, alto=60; frame=320×240");
    const overlay = dogCard.querySelector("svg");
    expect(overlay).toHaveAttribute("viewBox", "0 0 320 240");
    expect(overlay).toHaveAttribute("preserveAspectRatio", "xMidYMid meet");
    expect(overlay?.querySelector("rect")).toHaveAttribute("x", "12");
    expect(overlay?.querySelector("rect")).toHaveAttribute("y", "8");
    expect(overlay?.querySelector("rect")).toHaveAttribute("width", "80");
    expect(overlay?.querySelector("rect")).toHaveAttribute("height", "60");
    expect(screen.getByText(`2 capturas válidas · bucket ${bucket}`)).toBeInTheDocument();
  });

  it("presenta offsets distintos sin convertir las horas al huso local", async () => {
    const withPositiveOffset = {
      ...dog,
      captured_at: "2026-10-06T01:04:05.250+05:30",
      received_at: "2026-10-05T11:34:56-07:00",
    };
    vi.stubGlobal("fetch", vi.fn(async () => response([withPositiveOffset])));
    renderAt();

    const captured = await screen.findByText("6 oct 2026, 01:04:05.250 (UTC+05:30)");
    const received = screen.getByText("5 oct 2026, 11:34:56 (UTC-07:00)");
    expect(captured).toHaveAttribute("datetime", withPositiveOffset.captured_at);
    expect(received).toHaveAttribute("datetime", withPositiveOffset.received_at);
  });

  it("ordena por instante con offsets y desempata por capture_id sin mutar los datos", async () => {
    const tiedId = "00000000-0000-4000-8000-000000000001";
    const sameInstant = {
      ...dog,
      capture_id: tiedId,
      captured_at: cat.captured_at,
      image_key: `edge-captures/v1/images/${tiedId}.jpg`,
    };
    vi.stubGlobal("fetch", vi.fn(async () => response([cat, sameInstant, dog])));
    renderAt();

    expect(await screen.findByText(cat.capture_id)).toBeInTheDocument();
    expect(listItems()[0]).toHaveTextContent(dog.capture_id);
    expect(listItems()[1]).toHaveTextContent(tiedId);
    expect(listItems()[2]).toHaveTextContent(cat.capture_id);
  });

  it("navega desde el menú global a la ruta dentro de AppLayout", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response([])));
    renderAt("/dashboard");

    fireEvent.click(screen.getByRole("link", { name: "Capturas Edge" }));
    expect(await screen.findByRole("heading", { name: "Capturas Edge" })).toBeInTheDocument();
    expect(screen.getByText("No hay capturas edge")).toBeInTheDocument();
  });

  it("Actualizar consulta de nuevo y agrega la captura recién recibida", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response([cat]))
      .mockResolvedValueOnce(response([cat, dog]));
    vi.stubGlobal("fetch", fetcher);
    renderAt();

    expect(await screen.findByText(cat.capture_id)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Actualizar" }));
    expect(await screen.findByText(dog.capture_id)).toBeInTheDocument();
    expect(listItems()).toHaveLength(2);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("ignora una respuesta anterior que llega después de la actualización", async () => {
    const previous = deferredResponse();
    const latest = deferredResponse();
    const fetcher = vi
      .fn()
      .mockImplementationOnce(() => previous.promise)
      .mockImplementationOnce(() => latest.promise);
    vi.stubGlobal("fetch", fetcher);
    renderAt();

    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("status", { name: "Cargando capturas edge" })).toBeInTheDocument();
    const previousSignal = (fetcher.mock.calls[0]?.[1] as RequestInit | undefined)?.signal;
    fireEvent.click(screen.getByRole("button", { name: "Actualizar" }));
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(previousSignal?.aborted).toBe(true);

    await act(async () => latest.resolve(response([dog])));
    expect(await screen.findByText(dog.capture_id)).toBeInTheDocument();
    await act(async () => previous.resolve(response([cat])));
    expect(screen.getByText(dog.capture_id)).toBeInTheDocument();
    expect(screen.queryByText(cat.capture_id)).not.toBeInTheDocument();
  });

  it("explica un fallo de imagen sin atribuirle causa y recupera la URL al actualizar", async () => {
    const refreshed = { ...cat, image_url: "https://example.invalid/cat-refreshed.jpg" };
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(response([cat]))
      .mockResolvedValueOnce(response([refreshed]));
    vi.stubGlobal("fetch", fetcher);
    renderAt();

    const image = await screen.findByRole("img", { name: `Frame completo de la captura ${cat.capture_id}` });
    fireEvent.error(image);
    expect(screen.getByText("No se pudo cargar la imagen; la URL puede haber expirado.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Actualizar" }));
    await waitFor(() => {
      expect(screen.getByRole("img", { name: `Frame completo de la captura ${cat.capture_id}` })).toHaveAttribute("src", refreshed.image_url);
    });
    expect(screen.queryByText("No se pudo cargar la imagen; la URL puede haber expirado.")).not.toBeInTheDocument();
  });

  it("oculta el overlay si falla la imagen del frame", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response([dog])));
    renderAt();

    const image = await screen.findByRole("img", { name: `Frame completo de la captura ${dog.capture_id}` });
    expect(image.closest("article")?.querySelector("svg")).not.toBeNull();
    fireEvent.error(image);
    expect(screen.getByText("No se pudo cargar la imagen; la URL puede haber expirado.")).toBeInTheDocument();
    expect(screen.getByText(dog.capture_id).closest("article")?.querySelector("svg")).toBeNull();
  });

  it("valida el payload completo y rechaza metadatos inválidos", () => {
    expect(edgeCaptureListSchema.safeParse({ items: [cat, dog], total: 2, bucket }).success).toBe(true);
    expect(
      edgeCaptureListSchema.safeParse({ items: [{ ...cat, capture_id: "invalid" }], total: 1, bucket })
        .success
    ).toBe(false);
    expect(
      edgeCaptureListSchema.safeParse({ items: [{ ...dog, crop: { ...dog.crop, x: 300 } }], total: 1, bucket })
        .success
    ).toBe(false);
    expect(
      edgeCaptureListSchema.safeParse({ items: [{ ...cat, captured_at: "2026-02-30T12:00:00Z" }], total: 1, bucket })
        .success
    ).toBe(false);
  });
});
