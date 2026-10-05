import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ModelsPage } from "./Models";

/**
 * P3-15: la página lista las versiones del `registry.json` de P3-14 con su
 * estado S3, separa "versión del dataset" de "versión del modelo", muestra la
 * tarjeta + URL prefirmada y refleja el rechazo del backend al marcar una
 * versión activa sin objeto en S3. `fetch` va mockeado; no se toca ml-api real.
 */

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const PENDING = {
  status: "pending",
  ticket: "P3-14",
  message: "El catálogo de modelos todavía no existe; lo publica P3-14.",
};

function summary(overrides: Record<string, unknown> = {}) {
  return {
    version: "0.1.0",
    dataset_version: "v0.1.0",
    run_id: "b828e032156e423097e90d797b643faa",
    published_at: "2026-10-03T01:26:45Z",
    package_sha256: "a".repeat(64),
    registered_version_id: "v-old",
    selected: false,
    active: false,
    s3_status: { exists: true, version_id: "v-old", size_bytes: 1024, last_modified: null },
    ...overrides,
  };
}

function detail(overrides: Record<string, unknown> = {}) {
  return {
    status: "ready",
    version: "1.0.0",
    dataset_version: "v0.1.1",
    run_id: "7e7b4a4b35464cfebb6b41714a3ad931",
    published_at: "2026-10-03T01:27:42Z",
    package_sha256: "c".repeat(64),
    registered_version_id: "v-final",
    checkpoint_sha256: "e".repeat(64),
    run_kind: "campaign",
    manifest_id: "v0.1.1-53fc84fdaa07",
    selected: true,
    active: false,
    s3_bucket: "mlops-p3-models-222629887955",
    s3_key: "models/releases/v1.0.0/model_release_v1.0.0.tar.gz",
    card: "Modelo final, candidato r02.",
    download_url: "https://signed.example/models/releases/v1.0.0/model_release_v1.0.0.tar.gz",
    s3_status: { exists: true, version_id: "v-final", size_bytes: 2048, last_modified: null },
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("ModelsPage (P3-15)", () => {
  it("muestra que no hay versiones publicadas mientras falta el registry de P3-14", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(PENDING)));

    render(<ModelsPage />);

    expect(await screen.findByText("Sin versiones publicadas")).toBeInTheDocument();
    expect(screen.getByText(PENDING.message)).toBeInTheDocument();
  });

  it("separa la versión del dataset de la del modelo y marca la activa", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          status: "ready",
          active_version: "1.0.0",
          versions: [
            summary(),
            summary({
              version: "1.0.0",
              dataset_version: "v0.1.1",
              selected: true,
              active: true,
              s3_status: {
                exists: true,
                version_id: "v-final",
                size_bytes: 2048,
                last_modified: null,
              },
            }),
          ],
        })
      )
    );

    render(<ModelsPage />);

    expect(await screen.findByText(/Versión activa para inferencia: 1\.0\.0/)).toBeInTheDocument();
    expect(screen.getByText("dataset v0.1.0")).toBeInTheDocument();
    expect(screen.getByText("dataset v0.1.1")).toBeInTheDocument();
    expect(screen.getAllByText("activa").length).toBeGreaterThan(0);
    expect(screen.getByText("candidato")).toBeInTheDocument();
  });

  it("refleja el rechazo del backend al marcar una versión activa sin objeto en S3", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/models/active")) {
        expect(init?.method).toBe("POST");
        return Promise.resolve(
          jsonResponse(
            { error: "el objeto de la versión 1.0.0 no existe en S3; no se marca como activa" },
            400
          )
        );
      }
      if (url.endsWith("/models/1.0.0")) {
        return Promise.resolve(jsonResponse(detail()));
      }
      return Promise.resolve(
        jsonResponse({
          status: "ready",
          active_version: null,
          versions: [summary({ version: "1.0.0", dataset_version: "v0.1.1", selected: true })],
        })
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<ModelsPage />);

    fireEvent.click(await screen.findByText("1.0.0"));
    const activate = await screen.findByRole("button", { name: "Marcar como activa" });
    fireEvent.click(activate);

    expect(await screen.findByText(/no existe en S3; no se marca como activa/)).toBeInTheDocument();
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/ml-api/models/active",
        expect.objectContaining({ method: "POST" })
      )
    );
  });

  it("al marcar una versión activa refleja el cambio en la página", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/models/active")) {
        expect(init?.method).toBe("POST");
        return Promise.resolve(jsonResponse(detail({ active: true })));
      }
      if (url.endsWith("/models/1.0.0")) {
        return Promise.resolve(jsonResponse(detail()));
      }
      return Promise.resolve(
        jsonResponse({
          status: "ready",
          active_version: null,
          versions: [summary({ version: "1.0.0", dataset_version: "v0.1.1", selected: true })],
        })
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<ModelsPage />);

    fireEvent.click(await screen.findByText("1.0.0"));
    fireEvent.click(await screen.findByRole("button", { name: "Marcar como activa" }));

    expect(await screen.findByRole("button", { name: "Ya es la activa" })).toBeInTheDocument();
  });
});
