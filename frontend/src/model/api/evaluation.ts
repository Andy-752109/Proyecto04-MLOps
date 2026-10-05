import { ML_API_BASE_URL, mlApiRequest } from "./client";
import { evaluationResponseSchema } from "./contracts";

/** GET /ml-api/evaluation (P3-15): evaluación real, candado (P3-11) o pendiente (P3-13). */
export async function getEvaluation(signal?: AbortSignal) {
  return mlApiRequest("/evaluation", evaluationResponseSchema, { signal });
}

/**
 * URL del recorte real de un ejemplo (`GET /ml-api/crops/<crop_id>`). El
 * endpoint valida el formato del `crop_id` antes de servir el archivo de
 * `data/derived/crops/images/`, así que un valor inesperado simplemente da 404.
 */
export function cropImageUrl(cropId: string): string {
  return `${ML_API_BASE_URL}/crops/${encodeURIComponent(cropId)}`;
}

/** `predictions.csv` de P3-13, servido estático por nginx junto al resto de reportes. */
export const PREDICTIONS_CSV_URL = "/reports/evaluation/predictions.csv";
