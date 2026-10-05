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

// Espejo de `app/ml_api/contracts.py::PredictionResponse` (P3-16, #24).
export const predictionResponseSchema = z
  .object({
    predicted_label: z.string().min(1),
    probabilities: z.record(z.string(), z.number().min(0).max(1)),
    model_version: z.string().min(1),
    checkpoint_sha256: z.string().min(1),
  })
  .strict();

export const pendingEndpointSchema = z
  .object({
    status: z.literal("pending"),
    ticket: z.string().min(1),
    message: z.string().min(1),
  })
  .strict();

// --- P3-15: Evaluation y Models ---------------------------------------------

// P3-11: la selección no está cerrada (o el sello del test no coincide). Mismo
// contrato que `EvaluationLocked` en app/ml_api/contracts.py.
export const evaluationLockedSchema = z
  .object({
    status: z.literal("selection_not_closed"),
    ticket: z.literal("P3-11"),
    message: z.string().min(1),
  })
  .strict();

export const evaluationExampleSchema = z
  .object({
    crop_id: z.string().min(1),
    source_image_id: z.string().min(1),
    true_class: z.string().min(1),
    predicted_class: z.string().min(1),
    probability: z.number().finite(),
  })
  .strict();

export const evaluationClassMetricsSchema = z
  .object({
    precision: z.number().finite(),
    recall: z.number().finite(),
    f1: z.number().finite(),
    support: z.number().int().nonnegative(),
  })
  .strict();

// Espejo de `EvaluationReport` en app/ml_api/contracts.py: sello de P3-11 +
// reportes de P3-13, servidos por GET /ml-api/evaluation.
export const evaluationReportSchema = z
  .object({
    status: z.literal("ready"),
    run_id: z.string().min(1),
    release: z.string().min(1),
    manifest_id: z.string().min(1),
    manifest_sha256: z.string().min(1),
    checkpoint_sha256: z.string().min(1),
    selected_at: z.string().min(1),
    grid_row: z.string().min(1),
    best_val_accuracy: z.number().finite(),
    best_val_macro_f1: z.number().finite(),
    best_val_loss: z.number().finite(),
    classes: z.array(z.string().min(1)),
    accuracy: z.number().finite(),
    macro_f1: z.number().finite(),
    confusion_matrix: z.array(z.array(z.number().int().nonnegative())),
    per_class: z.record(z.string(), evaluationClassMetricsSchema),
    total: z.number().int().nonnegative(),
    baseline_majority_accuracy: z.number().finite(),
    most_confused_class: z.string().min(1),
    recall_per_class: z.record(z.string(), z.number().finite()),
    accuracy_hides_low_recall: z.boolean(),
    successes: z.array(evaluationExampleSchema),
    errors: z.array(evaluationExampleSchema),
  })
  .strict();

export const evaluationResponseSchema = z.discriminatedUnion("status", [
  evaluationReportSchema,
  evaluationLockedSchema,
  pendingEndpointSchema,
]);

export type EvaluationExample = z.infer<typeof evaluationExampleSchema>;
export type EvaluationReport = z.infer<typeof evaluationReportSchema>;
export type EvaluationResponse = z.infer<typeof evaluationResponseSchema>;

// Espejo de los contratos de Models en app/ml_api/contracts.py (P3-15). El
// catálogo real es `models/registry.json` (P3-14, #62): un dict `version ->
// {s3_path, sha256, VersionId, run_id, checkpoint_sha256, data_release,
// published_at}`. La versión del dataset (`dataset_version` = `data_release`)
// va separada de la del modelo (requisito del #23).
export const modelS3StatusSchema = z
  .object({
    exists: z.boolean(),
    version_id: z.string().nullable(),
    size_bytes: z.number().int().nullable(),
    last_modified: z.string().nullable(),
    // P3-15: la verificación en vivo de esta versión falló (red, SSO, permisos).
    error: z.string().nullable().default(null),
  })
  .strict();

export const modelSummarySchema = z
  .object({
    version: z.string().min(1),
    dataset_version: z.string().min(1),
    run_id: z.string().min(1),
    published_at: z.string().min(1),
    package_sha256: z.string().min(1),
    registered_version_id: z.string().min(1),
    selected: z.boolean(),
    active: z.boolean(),
    s3_status: modelS3StatusSchema,
  })
  .strict();

export const modelListSchema = z
  .object({
    status: z.literal("ready"),
    active_version: z.string().nullable(),
    versions: z.array(modelSummarySchema),
  })
  .strict();

export const modelListResponseSchema = z.discriminatedUnion("status", [
  modelListSchema,
  pendingEndpointSchema,
]);

export const modelDetailSchema = z
  .object({
    status: z.literal("ready"),
    version: z.string().min(1),
    dataset_version: z.string().min(1),
    run_id: z.string().min(1),
    published_at: z.string().min(1),
    package_sha256: z.string().min(1),
    registered_version_id: z.string().min(1),
    checkpoint_sha256: z.string().min(1),
    run_kind: z.string().nullable(),
    manifest_id: z.string().nullable(),
    selected: z.boolean(),
    active: z.boolean(),
    s3_bucket: z.string().min(1),
    s3_key: z.string().min(1),
    card: z.string().nullable(),
    download_url: z.string().nullable(),
    s3_status: modelS3StatusSchema,
  })
  .strict();

export type ModelSummary = z.infer<typeof modelSummarySchema>;
export type ModelListResponse = z.infer<typeof modelListResponseSchema>;
export type ModelDetail = z.infer<typeof modelDetailSchema>;
export type PredictionResponse = z.infer<typeof predictionResponseSchema>;
export type TrainingJob = z.infer<typeof trainingJobSchema>;
export type TrainingJobList = z.infer<typeof trainingJobListSchema>;
export type PendingEndpoint = z.infer<typeof pendingEndpointSchema>;
export type RunKind = z.infer<typeof runKindSchema>;
export type TrainingConfig = z.infer<typeof trainingConfigSchema>;
export type CreateTrainingJobRequest = z.infer<typeof createTrainingJobRequestSchema>;
