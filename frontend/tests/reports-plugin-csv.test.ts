// @vitest-environment node
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, expect, it } from "vitest";
import { readReport, reportsPlugin } from "../reports-plugin";

/**
 * P3-19: el servidor de desarrollo debe servir `predictions.csv` (enlace de
 * descarga de Evaluation) y nada más que ese CSV. Nginx ya lo sirve en Compose.
 *
 * Es un archivo aparte de `reports-plugin.test.ts` a propósito: aquel crea un
 * symlink en su `beforeEach`, que en Windows falla con EPERM sin permisos
 * especiales. Aquí no hay symlinks: la protección contra symlinks que escapan
 * de `reports/` sigue cubierta por `reports-plugin.test.ts` (en el CI).
 */

const CSV_URL = "/reports/evaluation/predictions.csv";
const CSV_BODY = "crop_id,true_class,predicted_class\n000275_000320,dog,cat\n";

let temp: string;
let root: string;
beforeEach(async () => {
  temp = await mkdtemp(path.join(tmpdir(), "p319-"));
  root = path.join(temp, "reports");
  await mkdir(path.join(root, "evaluation"), { recursive: true });
  await writeFile(path.join(root, "evaluation", "predictions.csv"), CSV_BODY);
  await writeFile(path.join(root, "evaluation", "otro.csv"), "secret\n");
  await writeFile(path.join(root, "predictions.csv"), "secret\n");
  await writeFile(path.join(root, "evaluation", "metrics.json"), "{}");
  await writeFile(path.join(temp, "private.csv"), "secret\n");
});
afterEach(async () => {
  await rm(temp, { recursive: true, force: true });
});

// Invoca el middleware HTTP real de Vite con dobles de request/response.
async function requestReport(url: string, method = "GET") {
  type Middleware = import("vite").Connect.NextHandleFunction;
  let handler: Middleware | undefined;
  const server = {
    middlewares: {
      use(middleware: Middleware) {
        handler = middleware;
      },
    },
  };
  const hook = reportsPlugin(root).configureServer;
  if (typeof hook !== "function") throw new Error("Expected configureServer hook");
  hook.call({} as never, server as unknown as import("vite").ViteDevServer);
  const middleware = handler;
  if (!middleware) throw new Error("HTTP middleware was not registered");
  return new Promise<{ status: number; body: string; headers: Record<string, string> }>(
    (resolve) => {
      const headers: Record<string, string> = {};
      const response = {
        statusCode: 200,
        setHeader(name: string, value: string) {
          headers[name] = value;
        },
        end(body?: Buffer | string) {
          resolve({ status: this.statusCode, body: body?.toString() ?? "", headers });
        },
      };
      middleware(
        { url, method } as import("node:http").IncomingMessage,
        response as unknown as import("node:http").ServerResponse,
        () => resolve({ status: 418, body: "next middleware", headers })
      );
    }
  );
}

it("GET sirve evaluation/predictions.csv como text/csv con su contenido", async () => {
  const result = await requestReport(CSV_URL);
  expect(result.status).toBe(200);
  expect(result.headers["Content-Type"]).toBe("text/csv; charset=utf-8");
  expect(result.body).toBe(CSV_BODY);
});

it("HEAD responde 200 text/csv sin cuerpo", async () => {
  const result = await requestReport(CSV_URL, "HEAD");
  expect(result.status).toBe(200);
  expect(result.headers["Content-Type"]).toBe("text/csv; charset=utf-8");
  expect(result.body).toBe("");
});

it("POST sobre el CSV sigue dando 405", async () => {
  expect((await requestReport(CSV_URL, "POST")).status).toBe(405);
});

it("un reporte JSON sigue sirviéndose como application/json", async () => {
  const result = await requestReport("/reports/evaluation/metrics.json");
  expect(result.status).toBe(200);
  expect(result.headers["Content-Type"]).toBe("application/json; charset=utf-8");
});

it.each([
  ["otro .csv en la misma carpeta", "/reports/evaluation/otro.csv"],
  ["predictions.csv fuera de evaluation/", "/reports/predictions.csv"],
  ["predictions.csv en otra subcarpeta", "/reports/other/evaluation/predictions.csv"],
  ["mayúsculas distintas", "/reports/evaluation/Predictions.csv"],
  ["barra final", "/reports/evaluation/predictions.csv/"],
  ["con .. hacia un .csv", "/reports/evaluation/../../private.csv"],
  ["con .. codificado hacia un .csv", "/reports/evaluation/%2e%2e/%2e%2e/private.csv"],
  ["con .. doblemente codificado", "/reports/%252e%252e/private.csv"],
  ["con .. que vuelve al CSV permitido", "/reports/evaluation/../evaluation/predictions.csv"],
])("rechaza con 404: %s", async (_name, url) => {
  const result = await requestReport(url);
  expect(result.status).toBe(404);
  expect(result.body).not.toContain("secret");
});

// Regresión: `evaluation/` es la única carpeta con un archivo no-JSON público. Si
// fuera un enlace hacia fuera de `reports/`, el `realpath` debe rechazarlo.
// `"junction"` es un directorio-junction en Windows (no pide permisos de
// administrador, a diferencia de un symlink) y se ignora en Linux/macOS, donde
// crea un symlink normal.
it("no sirve predictions.csv si evaluation/ es un enlace hacia fuera de reports/", async () => {
  const escapeTemp = await mkdtemp(path.join(tmpdir(), "p319-escape-"));
  try {
    const escapeRoot = path.join(escapeTemp, "reports");
    const outside = path.join(escapeTemp, "outside");
    await mkdir(escapeRoot, { recursive: true });
    await mkdir(outside, { recursive: true });
    await writeFile(path.join(outside, "predictions.csv"), "secret\n");
    await symlink(outside, path.join(escapeRoot, "evaluation"), "junction");

    await expect(readReport(escapeRoot, CSV_URL)).rejects.toThrow("Report outside root");
  } finally {
    await rm(escapeTemp, { recursive: true, force: true });
  }
});
