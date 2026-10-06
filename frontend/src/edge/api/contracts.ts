import { z } from "zod";

// Espejo del evento v1 de P4-02 y de EdgeCaptureList en app/ml_api/contracts.py.
// Los dos campos derivados pertenecen solo a la respuesta HTTP.
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const timestampSchema = z.iso.datetime({ offset: true });

const probabilitySchema = z.number().finite().min(0).max(1);

const cropSchema = z
  .object({
    x: z.number().int().nonnegative(),
    y: z.number().int().nonnegative(),
    width: z.number().int().positive(),
    height: z.number().int().positive(),
    frame_width: z.number().int().positive(),
    frame_height: z.number().int().positive(),
  })
  .strict()
  .refine(
    (crop) => crop.x + crop.width <= crop.frame_width && crop.y + crop.height <= crop.frame_height,
    "Recorte fuera del frame"
  );

export const edgeCaptureSchema = z
  .object({
    schema_version: z.literal("1"),
    capture_id: z.string().regex(UUID_V4),
    captured_at: timestampSchema,
    device_id: z.string().min(1),
    model_version: z.string().min(1),
    model_sha256: z.string().regex(/^[0-9a-fA-F]{64}$/),
    runtime: z.string().min(1),
    predicted_class: z.enum(["cat", "dog"]),
    confidence: probabilitySchema,
    probabilities: z.object({ cat: probabilitySchema, dog: probabilitySchema }).strict(),
    crop: cropSchema.nullable(),
    preprocess_ms: z.number().finite().nonnegative(),
    inference_ms: z.number().finite().nonnegative(),
    image_key: z.string().regex(/^edge-captures\/v1\/images\/[0-9a-f-]+\.jpg$/),
    received_at: timestampSchema,
    image_url: z.url(),
  })
  .strict()
  .refine((capture) => capture.image_key === `edge-captures/v1/images/${capture.capture_id}.jpg`, {
    path: ["image_key"],
    message: "image_key no corresponde a capture_id",
  });

export const edgeCaptureListSchema = z
  .object({
    items: z.array(edgeCaptureSchema),
    total: z.number().int().nonnegative(),
    bucket: z.string().min(1),
  })
  .strict();

export type EdgeCapture = z.infer<typeof edgeCaptureSchema>;
export type EdgeCaptureList = z.infer<typeof edgeCaptureListSchema>;
