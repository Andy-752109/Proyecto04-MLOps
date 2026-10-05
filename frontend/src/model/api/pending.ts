import { mlApiRequest } from "./client";
import { pendingEndpointSchema } from "./contracts";

/** Las 4 áreas de "Modelo" que todavía no tienen datos reales (ver docs/api-contratos.md). */
export async function getPending(path: "/experiments" | "/evaluation" | "/models" | "/inference") {
  return mlApiRequest(path, pendingEndpointSchema);
}
