import { z } from 'zod';

/**
 * Lo que manda el frontend tras un `POST /predict` exitoso (P3-16, #24) --
 * espejo de `PredictionResponse` (ml_api/contracts.py), salvo que aquí no
 * hace falta `predicted_label` por separado: ya es el campo que se valida.
 */
const inferenceSubmissionSchema = z.object({
  predictedLabel: z.string().min(1),
  probabilities: z.record(z.string(), z.number().min(0).max(1)),
  modelVersion: z.string().min(1),
  checkpointSha256: z.string().length(64),
});

export type InferenceSubmissionInput = z.infer<typeof inferenceSubmissionSchema>;

export function validateInferenceSubmission(input: unknown) {
  return inferenceSubmissionSchema.safeParse(input);
}
