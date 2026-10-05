import { mlApiRequest } from "./client";
import { modelDetailSchema, modelListResponseSchema } from "./contracts";

/** GET /ml-api/models (P3-15): catálogo de P3-14 o pendiente. */
export async function getModels(signal?: AbortSignal) {
  return mlApiRequest("/models", modelListResponseSchema, { signal });
}

/** GET /ml-api/models/{version}: tarjeta, procedencia y URL prefirmada. */
export async function getModel(version: string, signal?: AbortSignal) {
  return mlApiRequest(`/models/${encodeURIComponent(version)}`, modelDetailSchema, { signal });
}

/** POST /ml-api/models/active: el backend rechaza una versión sin objeto en S3. */
export async function setActiveModel(version: string) {
  return mlApiRequest("/models/active", modelDetailSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version }),
  });
}
