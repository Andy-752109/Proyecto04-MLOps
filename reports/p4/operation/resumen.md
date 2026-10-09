# Resumen de la corrida E2E

- Capturas de la corrida en el log local: 57; en S3 con evento igual al local: 57; en el portal con metadatos iguales: 57
- Paso 2 (≥20 capturas con fotos del PM): 20 capturas, 18 aciertos, 2 errores (90%); sin umbral de accuracy

| foto | real | predicho | confianza | `capture_id` |
|---|---|---|---|---|
| 2 | dog | cat | 0.778 | `d9bd109d-784e-426f-acb5-8743e44fef1e` |
| 16 | cat | dog | 0.910 | `d36937d6-5aa1-463b-8b79-d5365f2fa55e` |
