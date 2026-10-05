import { readFile, realpath } from "node:fs/promises";
import path from "node:path";
import type { Plugin } from "vite";

// P3-19: único archivo no-JSON público, el enlace de descarga de Evaluation.
const PREDICTIONS_CSV = "evaluation/predictions.csv";

function reportPathname(url: string): string {
  return decodeURIComponent(url.split("?")[0] ?? "");
}

/**
 * Serve only JSON reports (and predictions.csv), including release directories,
 * never symlink escapes.
 */
export async function readReport(root: string, url: string): Promise<Buffer> {
  const pathname = reportPathname(url);
  if (!pathname.startsWith("/reports/")) throw new Error("Invalid report path");
  const parts = pathname.slice("/reports/".length).split("/");
  if (parts.some((part) => !/^[A-Za-z0-9_-][A-Za-z0-9._-]*$/.test(part))) {
    throw new Error("Invalid report path");
  }
  if (!parts.at(-1)?.endsWith(".json") && parts.join("/") !== PREDICTIONS_CSV) {
    throw new Error("Only JSON reports and predictions.csv are public");
  }
  const base = await realpath(root);
  const file = await realpath(path.join(base, ...parts));
  if (!file.startsWith(`${base}${path.sep}`)) throw new Error("Report outside root");
  return readFile(file);
}

export function reportsPlugin(root: string): Plugin {
  return {
    name: "dataset-reports",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith("/reports/")) return next();
        if (req.method !== "GET" && req.method !== "HEAD") {
          res.statusCode = 405;
          res.end();
          return;
        }
        void readReport(root, req.url).then(
          (body) => {
            const isCsv = reportPathname(req.url ?? "") === `/reports/${PREDICTIONS_CSV}`;
            res.setHeader(
              "Content-Type",
              isCsv ? "text/csv; charset=utf-8" : "application/json; charset=utf-8"
            );
            res.setHeader("Cache-Control", "no-store");
            res.end(req.method === "HEAD" ? undefined : body);
          },
          () => {
            res.statusCode = 404;
            res.end("Report not found");
          }
        );
      });
    },
  };
}
