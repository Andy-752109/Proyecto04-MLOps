import type { TrainingConfig } from "./contracts";

/**
 * Espejo de `app/training/grid.py` -- rejilla de experimentos congelada
 * (docs/decisiones-proyecto3.md secciones 4, 6 y 7). Mismo criterio que los
 * contratos Pydantic/Zod: una sola fuente de verdad por lenguaje, nunca
 * inventada aparte.
 */

export const FROZEN_PATIENCE = 3;
export const FROZEN_MIN_DELTA = 0.001;

export const FROZEN_SEEDS = {
  seed_split: 42,
  seed_train: 43,
  seed_aug: 44,
  seed_model: 45,
} as const;

type GridRow = Omit<TrainingConfig, keyof typeof FROZEN_SEEDS | "patience" | "min_delta">;
type GridRowId = "r01" | "r02" | "r03" | "r04" | "r05" | "r06" | "r07" | "r08" | "r09" | "r10";

// r11/r12 no están definidos a propósito (sección 7): solo se deciden si
// val_accuracy de r01-r10 no llega a 0.90, no hay nada que autocompletar ahí.
// El tipo de la llave es explícito (no `Record<string, GridRow>`) para que
// `GRID.r01` resuelva seguro sin `| undefined` y una fila desconocida
// (`GRID[gridRow]` con texto libre del formulario) sí lo tenga.
export const GRID: Record<GridRowId, GridRow> = {
  r01: {
    optimizer: "adam",
    batch_size: 32,
    max_epochs: 15,
    learning_rate: 1e-3,
    image_size: 128,
    hidden_layers: 0,
    dropout: 0.0,
  },
  r02: {
    optimizer: "adam",
    batch_size: 32,
    max_epochs: 15,
    learning_rate: 3e-4,
    image_size: 128,
    hidden_layers: 1,
    dropout: 0.3,
  },
  r03: {
    optimizer: "adam",
    batch_size: 16,
    max_epochs: 15,
    learning_rate: 1e-3,
    image_size: 160,
    hidden_layers: 1,
    dropout: 0.3,
  },
  r04: {
    optimizer: "adam",
    batch_size: 16,
    max_epochs: 30,
    learning_rate: 3e-4,
    image_size: 160,
    hidden_layers: 0,
    dropout: 0.3,
  },
  r05: {
    optimizer: "sgd",
    batch_size: 32,
    max_epochs: 30,
    learning_rate: 1e-2,
    image_size: 128,
    hidden_layers: 1,
    dropout: 0.0,
  },
  r06: {
    optimizer: "sgd",
    batch_size: 16,
    max_epochs: 15,
    learning_rate: 1e-2,
    image_size: 160,
    hidden_layers: 0,
    dropout: 0.3,
  },
  r07: {
    optimizer: "sgd",
    batch_size: 32,
    max_epochs: 30,
    learning_rate: 3e-3,
    image_size: 160,
    hidden_layers: 1,
    dropout: 0.3,
  },
  r08: {
    optimizer: "adam",
    batch_size: 32,
    max_epochs: 30,
    learning_rate: 1e-4,
    image_size: 160,
    hidden_layers: 1,
    dropout: 0.0,
  },
  r09: {
    optimizer: "sgd",
    batch_size: 16,
    max_epochs: 30,
    learning_rate: 1e-2,
    image_size: 128,
    hidden_layers: 1,
    dropout: 0.3,
  },
  r10: {
    optimizer: "adam",
    batch_size: 16,
    max_epochs: 15,
    learning_rate: 1e-3,
    image_size: 128,
    hidden_layers: 1,
    dropout: 0.5,
  },
};

/** `GRID[row]` con `row` de texto libre (lo que escribe el formulario):
 * `undefined` para r11/r12 o cualquier texto que no sea una fila conocida. */
export function gridRowDefaults(row: string): GridRow | undefined {
  return (GRID as Record<string, GridRow | undefined>)[row];
}
