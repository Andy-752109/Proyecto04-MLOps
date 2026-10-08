# Comparación de calidad: original (P3) vs. INT8

Criterio 1.3. Meta: caída de accuracy ≤ 2 pp sobre **val (131)**. Test (73) se muestra aparte,
solo como referencia, y no se usó para decidir ni para ajustar la técnica.

## 1. Qué se comparó

| | Original | Optimizada |
|---|---|---|
| Modelo | P3 `1.0.0` (ResNet-18, PyTorch) | `1.0.0-int8` (ONNX, cuantización estática QDQ per-channel) |
| Runtime | PyTorch 2.14.0, 1 hilo (P4-01) | ONNX Runtime 1.30.0 CPU, 1 hilo, en laptop |
| SHA-256 | checkpoint `f759cde2…ab7d9` | `model_int8.onnx` `dffa9cf2…90392` |

Mismo manifiesto (`v0.1.1-53fc84fdaa07`) y mismo preprocesamiento (RGB → Resize 128×128 bilineal
con antialias → ToTensor → normalización ImageNet); la variante reutiliza `baseline.preprocessing`.
El runtime es el mismo (ONNX Runtime CPU) que usará el edge, pero se corrió en laptop; no se
midió en el dispositivo edge.

## 2. Resultados en val (conjunto principal)

| Métrica | Original | INT8 |
|---|---|---|
| Accuracy | 0.9618320610687023 (126/131) | 0.9618320610687023 (126/131) |
| F1 macro | 0.9618231625575566 | 0.9618231625575566 |
| **Caída de accuracy** | | **0.0 pp** (sin redondear) |

Matriz de confusión (filas = real, columnas = predicho, orden `cat, dog`), idéntica en ambos:
`[[64, 3], [2, 62]]`.

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| cat | 0.9697 | 0.9552 | 0.9624 | 67 |
| dog | 0.9538 | 0.9688 | 0.9612 | 64 |

**Decisión: cumple.** 0.0 pp ≤ 2 pp; no se escala a Karen/PM ni se activa el plan B de P4-04.

## 3. Errores relevantes y desacuerdos

Los 5 errores de val son los mismos en los dos modelos (los de P4-01); la cuantización no
corrigió ni introdujo ninguno. **Desacuerdos de clase entre modelos: 0 de 131.**

| crop_id | Real | Original / INT8 | prob_dog orig. → INT8 |
|---|---|---|---|
| 000135_000153 | cat | dog / dog | 0.871 → 0.853 |
| 000526_000521 | cat | dog / dog | 0.627 → 0.562 |
| 000633_000627 | cat | dog / dog | 0.644 → 0.575 |
| 000289_000331 | dog | cat / cat | 0.035 → 0.028 |
| 000476_000467 | dog | cat / cat | 0.101 → 0.069 |

Dos de los errores gato→perro (000526_000521, 000633_000627) son de baja confianza (prob_dog
0.56–0.64) y la cuantización los acerca un poco más al umbral de 0.5; no lo cruza ninguno. La
mayor diferencia de probabilidad por muestra en val es 0.0916 (|Δprob|), consistente con
`conversion_log.json`. En val ninguna muestra cambió de clase.

## 4. Calibración y separación de datos

La calibración usó 100 recortes **solo de train** (semilla 42, `calibration_ids.csv`, SHA-256
`e229786d…c4f3`), excluyendo también imágenes origen de val/test. Comprobado en
`compare_quality.py`: calibración ∩ val = ∅ y calibración ∩ test = ∅ (ambas listas vacías en
`metrics_*.json`, campo `calibration_overlap`).

## 5. Referencia secundaria: test (73), sin usar para decidir

| | Original | INT8 |
|---|---|---|
| Accuracy | 0.9863013698630136 (72/73) | 0.9863013698630136 |
| F1 macro | 0.9862081995087852 | 0.9862081995087852 |

Un solo error en ambos (000275_000320, perro→gato); 0 desacuerdos; |Δprob| máx. 0.0891. Las
predicciones del original en test se calcularon aquí con el checkpoint 1.0.0 (P4-01 solo dejó
val). Archivos: `comparison_test.csv`, `metrics_test.json`.

## 6. Verificación

- `comparison_val.csv`: 131 IDs únicos, iguales a los de val del manifiesto; ambas clases.
- Accuracy del original = la de P4-01 (0.9618320610687023, también `selection.json`).
- Recálculo independiente con scikit-learn (`recompute.sklearn_metrics`) coincide con
  `metrics_val.json` dentro de 1e-9: ver `reports/p4/quality/recompute_val.txt`.
- Las métricas de `metrics_*.json` se calculan en Python puro, para que el recálculo sea
  independiente.

Reproducir (desde `app/`, con `build/edge/1.0.0-int8/model_int8.onnx` y `eval_v1.0.0/` presentes):

```bash
uv run --group ml python -m edge_model.compare_quality
PYTHONPATH=. uv run --group ml python -m edge_model.recompute_quality
```

## 7. Limitaciones

Val tiene 131 muestras: un acierto de diferencia son 0.76 pp, así que la prueba solo descarta
caídas grandes. La medición fue en laptop, no en el edge. Las 20 capturas reales quedan fuera.
