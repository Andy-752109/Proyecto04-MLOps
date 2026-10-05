import { z } from "zod";

/**
 * Espejo campo a campo de `app/ml_api/contracts.py` (Pydantic) -- ver
 * docs/api-contratos.md. snake_case a propósito: este repo no mezcla
 * camelCase y snake_case entre payloads (mismo criterio que
 * lib/api/settings.ts).
 */

export const trainingJobStatusSchema = z.enum([
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
]);

// Contrato de MLflow (docs/decisiones-proyecto3.md sección 8, P3-08):
// "smoke" es el smoke test de P3-10, "campaign" una fila de la rejilla.
export const runKindSchema = z.enum(["smoke", "campaign"]);

export const trainingJobSchema = z
  .object({
    id: z.string().min(1),
    status: trainingJobStatusSchema,
    progress: z.number().finite().min(0).max(1),
    config: z.record(z.string(), z.unknown()),
    dataset_release: z.string().min(1),
    manifest_id: z.string().min(1),
    run_kind: runKindSchema,
    grid_row: z.string().min(1).nullable(),
    mlflow_run_id: z.string().min(1).nullable(),
    error: z.string().min(1).nullable(),
    logs: z.array(z.string()),
    heartbeat_at: z.string().min(1).nullable(),
  })
  .strict();

export const trainingJobListSchema = z
  .object({
    jobs: z.array(trainingJobSchema),
  })
  .strict();

// Espejo de `app/training/config.py` (TrainingConfig, Pydantic strict=True
// extra="forbid") -- la rejilla de hiperparámetros de docs/decisiones-proyecto3.md.
export const trainingConfigSchema = z
  .object({
    optimizer: z.enum(["adam", "sgd"]),
    batch_size: z.union([z.literal(16), z.literal(32)]),
    max_epochs: z.union([z.literal(15), z.literal(30)]),
    learning_rate: z.number().finite().min(1e-4).max(1e-2),
    image_size: z.union([z.literal(128), z.literal(160)]),
    hidden_layers: z.union([z.literal(0), z.literal(1)]),
    dropout: z.union([z.literal(0.0), z.literal(0.3), z.literal(0.5)]),
    seed_split: z.number().int(),
    seed_train: z.number().int(),
    seed_aug: z.number().int(),
    seed_model: z.number().int(),
    patience: z.number().int().min(0),
    min_delta: z.number().finite().min(0),
  })
  .strict();

// Mismas reglas que `app/ml_api/training_jobs.py` (GRID_ROW_PATTERN) y
// `app/training/tracking.py` (P3-08): grid_row vacío en smoke, r01-r12 en
// campaign. Repetidas aquí a propósito -- un valor inválido no debe generar
// ni un solo request (ver checklist de P3-09).
const GRID_ROW_PATTERN = /^r(0[1-9]|1[0-2])$/;

export const createTrainingJobRequestSchema = z
  .object({
    dataset_release: z.string().min(1),
    config: trainingConfigSchema,
    run_kind: runKindSchema,
    grid_row: z.string().min(1).nullable(),
  })
  .strict()
  .refine((request) => request.run_kind !== "smoke" || request.grid_row === null, {
    message: "grid_row debe estar vacío para run_kind 'smoke'",
    path: ["grid_row"],
  })
  .refine(
    (request) =>
      request.run_kind !== "campaign" ||
      (request.grid_row !== null && GRID_ROW_PATTERN.test(request.grid_row)),
    { message: "grid_row debe ser r01–r12 para run_kind 'campaign'", path: ["grid_row"] }
  );

export const pendingEndpointSchema = z
  .object({
    status: z.literal("pending"),
    ticket: z.string().min(1),
    message: z.string().min(1),
  })
  .strict();

export type TrainingJob = z.infer<typeof trainingJobSchema>;
export type TrainingJobList = z.infer<typeof trainingJobListSchema>;
export type PendingEndpoint = z.infer<typeof pendingEndpointSchema>;
export type RunKind = z.infer<typeof runKindSchema>;
export type TrainingConfig = z.infer<typeof trainingConfigSchema>;
export type CreateTrainingJobRequest = z.infer<typeof createTrainingJobRequestSchema>;
