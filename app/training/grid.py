"""Rejilla de experimentos congelada (P3-01, `docs/decisiones-proyecto3.md`
secciones 4, 6 y 7). Fuente de verdad única -- `frontend/src/model/api/grid.ts`
es su espejo TS, igual criterio que los contratos Pydantic/Zod.

`GRID` no trae `r11`/`r12`: la sección 7 los deja sin definir a propósito,
condicionados a que `val_accuracy` de r01-r10 no llegue a 0.90 -- se deciden
ad-hoc si hace falta, no se puede validar contra una fila que no existe
todavía.
"""

# Sección 4: igual para cualquier corrida, no solo para la rejilla.
FROZEN_PATIENCE = 3
FROZEN_MIN_DELTA = 0.001

# Sección 6: igual para cualquier corrida.
FROZEN_SEEDS = {
    "seed_split": 42,
    "seed_train": 43,
    "seed_aug": 44,
    "seed_model": 45,
}

# Sección 7. Los 7 parámetros que varían por fila; patience/min_delta/semillas
# son los de arriba, iguales en las 10.
GRID: dict[str, dict] = {
    "r01": {
        "optimizer": "adam",
        "batch_size": 32,
        "max_epochs": 15,
        "learning_rate": 1e-3,
        "image_size": 128,
        "hidden_layers": 0,
        "dropout": 0.0,
    },
    "r02": {
        "optimizer": "adam",
        "batch_size": 32,
        "max_epochs": 15,
        "learning_rate": 3e-4,
        "image_size": 128,
        "hidden_layers": 1,
        "dropout": 0.3,
    },
    "r03": {
        "optimizer": "adam",
        "batch_size": 16,
        "max_epochs": 15,
        "learning_rate": 1e-3,
        "image_size": 160,
        "hidden_layers": 1,
        "dropout": 0.3,
    },
    "r04": {
        "optimizer": "adam",
        "batch_size": 16,
        "max_epochs": 30,
        "learning_rate": 3e-4,
        "image_size": 160,
        "hidden_layers": 0,
        "dropout": 0.3,
    },
    "r05": {
        "optimizer": "sgd",
        "batch_size": 32,
        "max_epochs": 30,
        "learning_rate": 1e-2,
        "image_size": 128,
        "hidden_layers": 1,
        "dropout": 0.0,
    },
    "r06": {
        "optimizer": "sgd",
        "batch_size": 16,
        "max_epochs": 15,
        "learning_rate": 1e-2,
        "image_size": 160,
        "hidden_layers": 0,
        "dropout": 0.3,
    },
    "r07": {
        "optimizer": "sgd",
        "batch_size": 32,
        "max_epochs": 30,
        "learning_rate": 3e-3,
        "image_size": 160,
        "hidden_layers": 1,
        "dropout": 0.3,
    },
    "r08": {
        "optimizer": "adam",
        "batch_size": 32,
        "max_epochs": 30,
        "learning_rate": 1e-4,
        "image_size": 160,
        "hidden_layers": 1,
        "dropout": 0.0,
    },
    "r09": {
        "optimizer": "sgd",
        "batch_size": 16,
        "max_epochs": 30,
        "learning_rate": 1e-2,
        "image_size": 128,
        "hidden_layers": 1,
        "dropout": 0.3,
    },
    "r10": {
        "optimizer": "adam",
        "batch_size": 16,
        "max_epochs": 15,
        "learning_rate": 1e-3,
        "image_size": 128,
        "hidden_layers": 1,
        "dropout": 0.5,
    },
}
