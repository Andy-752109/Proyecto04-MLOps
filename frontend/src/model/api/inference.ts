import { mlApiRequest } from "./client";
import { predictionResponseSchema } from "./contracts";

// Mismo límite que `app/ml_api/inference.py::MAX_UPLOAD_SIZE_BYTES` (P3-16, #24).
export const MAX_PREDICT_IMAGE_SIZE_BYTES = 5 * 1024 * 1024;
export const ALLOWED_PREDICT_MIME_TYPES = ["image/jpeg", "image/png"] as const;

/** Validación de cliente: no reemplaza la de `ml-api`, es feedback inmediato
 * (mismo criterio que `lib/api/images.ts::validateImageFile`). */
export function validatePredictImageFile(file: File): string | null {
  if (
    !ALLOWED_PREDICT_MIME_TYPES.includes(file.type as (typeof ALLOWED_PREDICT_MIME_TYPES)[number])
  ) {
    return "Tipo de archivo no soportado. Usa JPEG o PNG.";
  }
  if (file.size > MAX_PREDICT_IMAGE_SIZE_BYTES) {
    return "Excede el límite de 5 MB.";
  }
  return null;
}

export function predictFromImage(file: File) {
  const body = new FormData();
  body.append("image", file);
  return mlApiRequest("/predict", predictionResponseSchema, { method: "POST", body });
}

export function predictFromCrop(imageId: number, annotationId: number) {
  const body = new FormData();
  body.append("image_id", String(imageId));
  body.append("annotation_id", String(annotationId));
  return mlApiRequest("/predict", predictionResponseSchema, { method: "POST", body });
}
