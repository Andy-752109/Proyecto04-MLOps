# Entrega P3 — guion de demo y hallazgos (P3-18)

Calendario: congelamiento de funcionalidades **jue 1, 18:00** · clon limpio **vie 2, 9:00** ·
autoevaluación **11:00** · demo y entrega en la tarde.

## Guion de la demo (cronometrado)

| # | Min | Acción | Qué debe verse |
|---|---|---|---|
| 1 | 1 | `/model/training`: `smoke` → "Lanzar entrenamiento" → recargar la página | el trabajo sigue ahí con su progreso (MariaDB) |
| 2 | 1 | Copiar el `run_id` y abrirlo en <http://localhost:5050> | la misma corrida en MLflow |
| 3 | 1 | `/evaluation` | accuracy 0.9863 (72/73), matriz de confusión, baseline 0.5342 |
| 4 | 1 | `/models`: abrir `1.0.0` → "Marcar como activa"; luego `0.1.0` | SHA-256, estado en S3, versión activa cambia |
| 5 | 1 | `/inference`, imagen nueva | `model_version` y `checkpoint_sha256` de la versión activa |
| 6 | 1 | `/inference`, recorte | clase y probabilidades; cambiar versión y repetir: el SHA cambia |
| 7 | 0.5 | Subir un archivo inválido (p. ej. `.txt`) | rechazo con mensaje claro |
| 8 | 0.5 | "Enviar a cola de anotación" (imagen nueva) | "Enviada a la cola (imagen #N)" |

Antes de empezar: `aws sso login --profile mlops-p3`, `docker compose up`, y una predicción de
calentamiento (la primera baja el paquete de S3). Dejar la versión activa en `1.0.0`.

## Cierre

- [ ] CI en verde en el PR
- [ ] `git status` limpio y `dvc status` → "up to date" (repetir `dvc repro` tras mergear P3-17)
- [ ] Tag `p3-entrega` en el commit de la demo
- [ ] Entrega por el medio que indicó el profesor

## Hallazgos del clon limpio (llenar por dueño)

| # | Paso del README | Qué falló | Dueño | Corregido |
|---|---|---|---|---|
| 1 | | | | ☐ |

## Autoevaluación

Prompt del profesor completo sobre el clon. Meta: **≥ 95** con M1–M4 en CUMPLE.

| Punto perdido | Corrección | Dueño | Hecho |
|---|---|---|---|
| | | | ☐ |
