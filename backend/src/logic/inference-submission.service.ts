import { createInferenceSubmissionRow, findImageById } from '../data/index.js';
import { NotFoundError, ValidationError } from './errors.js';
import { validateInferenceSubmission } from './inference-submission.validation.js';

export interface CreateInferenceSubmissionResult {
  id: number;
}

/**
 * "Enviar a cola de anotación" (P3-16, #24): la imagen ya se creó con
 * `uploadImage` (estado `pending` por default, image-upload.service.ts) --
 * esto solo adjunta la sugerencia del modelo a esa imagen, ya existente.
 */
export async function createInferenceSubmission(
  imageId: number,
  body: unknown,
): Promise<CreateInferenceSubmissionResult> {
  const image = await findImageById(imageId);

  if (!image) {
    throw new NotFoundError('La imagen no existe.');
  }

  const parsed = validateInferenceSubmission(body);

  if (!parsed.success) {
    throw new ValidationError('La sugerencia de inferencia no es válida.');
  }

  const id = await createInferenceSubmissionRow({
    imageId,
    predictedLabel: parsed.data.predictedLabel,
    probabilities: parsed.data.probabilities,
    modelVersion: parsed.data.modelVersion,
    checkpointSha256: parsed.data.checkpointSha256,
  });

  return { id };
}
