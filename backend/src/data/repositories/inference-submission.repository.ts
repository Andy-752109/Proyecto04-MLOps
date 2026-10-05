import { db } from '../db/client.js';
import { inferenceSubmissions, type NewInferenceSubmissionRow } from '../db/schema.js';

/**
 * Guarda en MariaDB la sugerencia del modelo sobre una imagen ya creada
 * (P3-16, #24). No valida que `imageId` exista -- eso lo hace la capa
 * Logic antes de llamar aquí, igual que el resto del repositorio.
 */
export async function createInferenceSubmissionRow(
  submission: NewInferenceSubmissionRow,
): Promise<number> {
  const result = await db.insert(inferenceSubmissions).values(submission).$returningId();

  const created = result[0];

  if (!created) {
    throw new Error('No se pudo guardar la sugerencia de inferencia en MariaDB.');
  }

  return created.id;
}
