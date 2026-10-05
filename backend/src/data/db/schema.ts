import { relations } from 'drizzle-orm';
import {
  bigint,
  boolean,
  double,
  index,
  int,
  json,
  mysqlEnum,
  mysqlTable,
  text,
  timestamp,
  uniqueIndex,
  varchar,
} from 'drizzle-orm/mysql-core';

/**
 * Guarda los metadatos de las imágenes.
 * El archivo real se almacena en MinIO mediante storageKey.
 */

export const images = mysqlTable(
  'images',
  {
    // ID único de la imagen.
    id: bigint('id', {
      mode: 'number',
      unsigned: true,
    })
      .autoincrement()
      .primaryKey(),

    // Nombre original del archivo.
    filename: varchar('filename', {
      length: 255,
    }).notNull(),

    // Ruta o key del archivo dentro de MinIO.
    storageKey: varchar('storage_key', {
      length: 512,
    }).notNull(),

    // Tipo de archivo, por ejemplo image/jpeg.
    mimeType: varchar('mime_type', {
      length: 100,
    }).notNull(),

    // Dimensiones de la imagen en píxeles.
    width: int('width', {
      unsigned: true,
    }).notNull(),

    height: int('height', {
      unsigned: true,
    }).notNull(),

    // Tamaño del archivo en bytes.
    sizeBytes: bigint('size_bytes', {
      mode: 'number',
      unsigned: true,
    }).notNull(),

    // Estado actual del proceso de anotación.
    status: mysqlEnum('status', ['pending', 'in_progress', 'completed'])
      .notNull()
      .default('pending'),

    // Fecha de creación del registro.
    createdAt: timestamp('created_at').notNull().defaultNow(),

    // Fecha de la última actualización.
    updatedAt: timestamp('updated_at').notNull().defaultNow().onUpdateNow(),
  },

  // Índices para mejorar búsquedas y filtros.
  (table) => [
    uniqueIndex('images_storage_key_unique').on(table.storageKey),
    index('images_status_idx').on(table.status),
    index('images_created_at_idx').on(table.createdAt),
    index('images_status_created_at_idx').on(table.status, table.createdAt),
  ],
);
/**
 * categories
 *
 * Categorías/clases usadas para anotar imágenes (equivalente a
 * `categories` en el formato COCO).
 */

export const categories = mysqlTable(
  'categories',
  {
    // ID único de la categoría.
    id: bigint('id', {
      mode: 'number',
      unsigned: true,
    })
      .autoincrement()
      .primaryKey(),

    // Nombre de la categoría, por ejemplo person o car.
    name: varchar('name', {
      length: 150,
    }).notNull(),

    // Color usado para mostrar la bounding box en la interfaz (hexadecimal).
    color: varchar('color', {
      length: 7,
    }).notNull(),

    // Fecha de creación de la categoría.
    createdAt: timestamp('created_at').notNull().defaultNow(),
  },

  // Evita que existan dos categorías con el mismo nombre.
  (table) => [uniqueIndex('categories_name_unique').on(table.name)],
);

/**
 * annotations
 *
 * Bounding box asociado a una imagen y una categoría. Los campos de bbox
 * siguen la convención de COCO (x, y = esquina superior izquierda;
 * width/height = dimensiones de la caja) para que la futura fase de
 * exportación a COCO JSON no requiera cambios de schema.
 */
export const annotations = mysqlTable(
  'annotations',
  {
    // ID único de la anotación.
    id: bigint('id', {
      mode: 'number',
      unsigned: true,
    })
      .autoincrement()
      .primaryKey(),

    // Imagen a la que pertenece la bounding box.
    imageId: bigint('image_id', {
      mode: 'number',
      unsigned: true,
    })
      .notNull()
      .references(() => images.id, {
        onDelete: 'cascade',
      }),

    // Categoría asignada a la bounding box.
    categoryId: bigint('category_id', {
      mode: 'number',
      unsigned: true,
    })
      .notNull()
      .references(() => categories.id, {
        onDelete: 'restrict',
      }),

    // Posición y tamaño de la caja en píxeles.
    bboxX: double('bbox_x').notNull(),
    bboxY: double('bbox_y').notNull(),
    bboxWidth: double('bbox_width').notNull(),
    bboxHeight: double('bbox_height').notNull(),

    // Área de la bounding box para la exportación COCO.
    area: double('area').notNull(),

    // Campo requerido por el formato COCO.
    isCrowd: boolean('iscrowd').notNull().default(false),

    // Fecha de creación de la anotación.
    createdAt: timestamp('created_at').notNull().defaultNow(),

    // Fecha de la última modificación.
    updatedAt: timestamp('updated_at').notNull().defaultNow().onUpdateNow(),
  },

  // Índices para buscar anotaciones por imagen y categoría.
  (table) => [
    index('annotations_image_id_idx').on(table.imageId),
    index('annotations_category_id_idx').on(table.categoryId),
    index('annotations_image_category_idx').on(table.imageId, table.categoryId),
  ],
);

/**
 * Relación: una imagen puede tener muchas anotaciones.
 */
export const imagesRelations = relations(images, ({ many }) => ({
  annotations: many(annotations),
}));

/**
 * Relación: una categoría puede pertenecer a muchas anotaciones.
 */
export const categoriesRelations = relations(categories, ({ many }) => ({
  annotations: many(annotations),
}));

/**
 * Cada anotación pertenece a una imagen y a una categoría.
 */
export const annotationsRelations = relations(annotations, ({ one }) => ({
  image: one(images, {
    fields: [annotations.imageId],
    references: [images.id],
  }),

  category: one(categories, {
    fields: [annotations.categoryId],
    references: [categories.id],
  }),
}));

/**
 * training_jobs (P3-03)
 *
 * Estado de una corrida de entrenamiento de la campaña de experimentos.
 * `trainer-worker` (Python) es quien escribe: crea la fila al arrancar una
 * corrida y va actualizando `status`/`progress`/`logs`/`heartbeatAt`
 * mientras entrena; `ml-api` (Python) solo lee, para la página Training.
 * `id` es un identificador propio de la app (ej. "r01"), NO el run id que
 * asigna MLflow -- ese vive aparte en `mlflowRunId`, porque una corrida
 * podría reintentarse con un `mlflowRunId` nuevo sin cambiar de fila.
 */
export const trainingJobs = mysqlTable(
  'training_jobs',
  {
    // Identificador de la corrida (ej. "r01"), no autoincremental: lo asigna
    // quien encola el job, antes de que exista la fila.
    id: varchar('id', {
      length: 64,
    }).primaryKey(),

    status: mysqlEnum('status', ['queued', 'running', 'completed', 'failed', 'cancelled'])
      .notNull()
      .default('queued'),

    // Fracción completada de epocas_maximas, 0-1.
    progress: double('progress').notNull().default(0),

    // Hiperparámetros de la corrida (optimizador, batch, lr, ...). La forma
    // exacta la define P3-07 (entrenador); aquí solo se guarda tal cual.
    config: json('config').notNull(),

    // Versión del release de datos usada (ej. "v0.1.1"). Nombrada
    // "dataset_release", no "release": es palabra reservada en SQL y rompe
    // cualquier consulta cruda sin comillas (probado con MariaDB real).
    datasetRelease: varchar('dataset_release', {
      length: 100,
    }).notNull(),

    // Manifiesto congelado (P3-06) sobre el que corrió esta corrida.
    manifestId: varchar('manifest_id', {
      length: 100,
    }).notNull(),

    // Contrato de MLflow (docs/decisiones-proyecto3.md sección 8, P3-08):
    // 'smoke' (P3-10 punto 1) o 'campaign' (una fila de la rejilla de P3-01).
    // Van en columnas propias, no dentro de `config`, porque no son
    // hiperparámetros del modelo -- son procedencia de la corrida, y
    // `training.TrainingConfig` es `extra="forbid"` (P3-05): meterlos ahí
    // rompería esa validación en vez de la suya propia.
    runKind: mysqlEnum('run_kind', ['smoke', 'campaign']).notNull(),

    // Fila de la rejilla de P3-01 (ej. "r01"), solo para run_kind='campaign';
    // null en 'smoke'. Mismo patrón r0[1-9]|r1[0-2] que valida tracking.py.
    gridRow: varchar('grid_row', {
      length: 10,
    }),

    // Null hasta que MLflow crea el run real.
    mlflowRunId: varchar('mlflow_run_id', {
      length: 100,
    }),

    // Mensaje de error si status = 'failed'; null en cualquier otro caso.
    error: text('error'),

    // Líneas de log recientes, más nuevas al final. No es el log completo
    // (eso vive en MLflow/artefactos): solo lo suficiente para la UI.
    logs: json('logs').notNull(),

    // Último "sigo vivo" de trainer-worker. Null antes de arrancar; una
    // corrida sin heartbeat reciente y status='running' se considera colgada.
    heartbeatAt: timestamp('heartbeat_at'),

    // P3-09 (revisión de Uriel): ml-api pone esta marca cuando alguien pide
    // cancelar; trainer-worker la revisa entre mensajes de progreso (no hay
    // forma de interrumpirlo al instante) y, al verla, mata el subproceso y
    // deja el job en 'cancelled'. Null mientras nadie la pidió.
    cancelRequestedAt: timestamp('cancel_requested_at'),

    createdAt: timestamp('created_at').notNull().defaultNow(),
    updatedAt: timestamp('updated_at').notNull().defaultNow().onUpdateNow(),
  },

  (table) => [
    index('training_jobs_status_idx').on(table.status),
    index('training_jobs_heartbeat_at_idx').on(table.heartbeatAt),
  ],
);

export type TrainingJobRow = typeof trainingJobs.$inferSelect;
export type NewTrainingJobRow = typeof trainingJobs.$inferInsert;

/**
 * inference_submissions (P3-16, #24)
 *
 * "Enviar a cola de anotación" desde la página Inference: la imagen en sí
 * ya se creó con el `POST /images` que existe desde P2 (`image-upload
 * .service.ts`, estado `pending` por default) -- esta tabla solo guarda la
 * sugerencia del modelo sobre esa imagen, para que quien anote la vea sin
 * tener que volver a correr la inferencia. `ml-api` (Python) no escribe
 * aquí directamente: el backend Node crea la fila después de llamar a
 * `POST /predict` él mismo.
 */
export const inferenceSubmissions = mysqlTable(
  'inference_submissions',
  {
    id: bigint('id', {
      mode: 'number',
      unsigned: true,
    })
      .autoincrement()
      .primaryKey(),

    // Imagen creada por este mismo flujo de envío a cola.
    imageId: bigint('image_id', {
      mode: 'number',
      unsigned: true,
    })
      .notNull()
      .references(() => images.id, {
        onDelete: 'cascade',
      }),

    // "cat"/"dog" -- la clase que predijo el modelo, no la que anote la persona.
    predictedLabel: varchar('predicted_label', {
      length: 50,
    }).notNull(),

    // Probabilidad por clase ({"cat": 0.9, "dog": 0.1}), igual que devuelve
    // `POST /predict` -- conserva el detalle completo, no solo la ganadora.
    probabilities: json('probabilities').notNull(),

    // Versión y hash del modelo que hizo la sugerencia (ver `models/registry.json`,
    // P3-14) -- trazabilidad: con qué modelo exacto se sugirió esta clase.
    modelVersion: varchar('model_version', {
      length: 50,
    }).notNull(),

    checkpointSha256: varchar('checkpoint_sha256', {
      length: 64,
    }).notNull(),

    createdAt: timestamp('created_at').notNull().defaultNow(),
  },

  (table) => [index('inference_submissions_image_id_idx').on(table.imageId)],
);

export type InferenceSubmissionRow = typeof inferenceSubmissions.$inferSelect;
export type NewInferenceSubmissionRow = typeof inferenceSubmissions.$inferInsert;

/**
 * Tipos TypeScript generados automáticamente desde el esquema.
 */
export type Image = typeof images.$inferSelect;
export type NewImage = typeof images.$inferInsert;

export type Category = typeof categories.$inferSelect;
export type NewCategory = typeof categories.$inferInsert;

export type Annotation = typeof annotations.$inferSelect;
export type NewAnnotation = typeof annotations.$inferInsert;
