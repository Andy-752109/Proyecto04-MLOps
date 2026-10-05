# Evaluación final (P3-13 / #20)

Evaluación **única** del candidato `r02` sobre el split `test`, hecha después de
sellar la selección (`reports/selection.json`, ver `docs/seleccion-candidato.md`).
Se corrió una sola vez; `app/final.py` se niega a repetirla.

- **Candidato:** `r02` · `run_id` `7e7b4a4b35464cfebb6b41714a3ad931`
- **`checkpoint_sha256`:** `f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9` (verificado antes de predecir)
- **Manifiesto:** `v0.1.1` · split `test`, **73 recortes**
- **Dónde quedaron las métricas en MLflow:** dentro de la corrida de `r02` (`7e7b4a4b35464cfebb6b41714a3ad931`), con prefijo `test_*`. No hay corrida de evaluación aparte (ver «MLflow: dónde están los resultados»).
- **Comando:** `app/.venv/bin/python app/final.py --split test --checkpoint <best.pt local>`

## Métricas

| Métrica | Valor |
|---|---|
| `accuracy` (diagonal ÷ total, sin redondear) | **0.9863013698630136** (72 / 73) |
| `macro_f1` | 0.9862081995087851 |
| Baseline de clase mayoritaria | 0.5342 |
| Accuracy en validación (r02) | 0.9618 |

| Clase | Precisión | Recall | F1 | Support |
|---|---|---|---|---|
| cat | 0.9750 | 1.0000 | 0.9873 | 39 |
| dog | 1.0000 | 0.9706 | 0.9851 | 34 |

Matriz de confusión (filas = clase real, columnas = predicha):

| | pred. cat | pred. dog |
|---|---|---|
| **real cat** | 39 | 0 |
| **real dog** | 1 | 33 |

`recompute.py` (scikit-learn, desde `predictions.csv`) coincide con `metrics.json`.

## Análisis de errores

- **Clase más confundida:** `dog`, con 1 error (un perro clasificado como gato).
- **Único error:** recorte `000275_000320` (imagen fuente `275`): real `dog`,
  predicho `cat` con probabilidad 0.577, es decir, una decisión poco segura
  (`prob_dog` = 0.423).
- **¿La accuracy oculta un recall bajo?** No: ninguna clase baja de 0.70
  (el menor es `dog`, 0.9706).
- **Aciertos:** los 6 primeros tienen probabilidad entre 0.95 y 1.00
  (`analysis.json`, campo `successes`).

## Cómo leer el resultado

- Supera con holgura el baseline (0.534) y el objetivo de 0.85.
- El `test` solo tiene 73 recortes: un error mueve la accuracy 1.4 puntos. Un
  intervalo de Wilson al 95 % para 72/73 va de ≈0.926 a ≈0.998, así que el
  0.986 no es una estimación precisa. La diferencia con validación (0.962,
  131 recortes) **no es significativa**.
- Con un solo error no hay un patrón de fallo que analizar; solo se puede decir
  que fue un caso de baja confianza.
- El test se usó una vez y no se ajustó nada después de verlo.

## Archivos

`reports/evaluation/`: `predictions.csv` (`crop_id`, `source_image_id`, clase real
y predicha, probabilidades), `metrics.json` y `analysis.json`. Los tres se
recalculan con `app/recompute.py`.

## Nota de reproducción

El proxy de artefactos de MLflow falló al bajar el checkpoint (~45 MB) por HTTP,
así que `best.pt` se reconstruyó desde el almacenamiento de MinIO y se pasó con
`--checkpoint`. `final.py` verifica igualmente su SHA-256 contra `selection.json`.

## MLflow: dónde están los resultados (para humanos y agentes)

**Regla.** Las métricas de test viven **dentro de la corrida del candidato (`r02`)**, no
en una corrida aparte (indicación del profe, 2 de octubre de 2026).

| Qué | Dónde |
|---|---|
| Corrida | `7e7b4a4b35464cfebb6b41714a3ad931` (`grid_row=r02`), experimento `clasificador-perro-gato` |
| Métricas | `test_accuracy` 0.9863013698630136 · `test_macro_f1` 0.9862081995087851 · `test_baseline_accuracy` 0.5342465753424658 · `test_num_crops` 73 |
| Artefactos | carpeta `evaluation/`: `predictions.csv`, `metrics.json`, `analysis.json` (los mismos de `reports/evaluation/`) |
| Tags de la evaluación | `eval_split=test`, `evaluation_started_at=2026-10-02T13:38:41+00:00` |

**Cuándo se permite `test_*`.** `app/selection/validate.py` y `app/selection/select.py`
rechazan cualquier métrica `test_*` salvo en la corrida que tenga `selected_candidate=true`
**y** `selected_at` (`RunSummary.is_selected_candidate`, PR #59). En cualquier otra
corrida, antes o después del candado, siguen prohibidas. `reports/experiments_validity.json`
sigue en 10/10 con las métricas dentro de r02.

**Unicidad.** `app/final.py --split test` se niega a repetirse si existe
`reports/evaluation/metrics.json` o si el candidato ya tiene alguna métrica `test_*`.

### Historial de este cambio (2 de octubre de 2026)

1. `final.py` (PR #56) creó una corrida aparte `0bb3379f5a444c00b2627e8135d3eb34`
   copiando **todos** los tags de r02. Eso dejó `selected_candidate=true`,
   `run_kind=campaign`, `grid_row=r02` y el mismo nombre que r02 en una corrida que no era
   el candidato.
2. Se corrigieron esos tags a mano en esa corrida (`selected_candidate`, `run_kind`
   y `grid_row` eliminados; nombre `evaluacion-final-r02`; `evaluated_grid_row=r02`).
   Las métricas y los archivos no se tocaron.
3. El profe indicó que las métricas de test deben ir en la misma corrida de r02. El PR #59
   cambió `final.py`, `validate.py` y `select.py` para eso.
4. Las métricas ya existentes (misma evaluación, sin reevaluar) se registraron en r02 con
   los mismos valores que tenía la corrida aparte, comprobados antes de escribir.
5. La corrida `0bb3379f…` se marcó como **eliminada** (borrado suave de MLflow, restaurable:
   `MlflowClient().restore_run("0bb3379f5a444c00b2627e8135d3eb34")`). Sus archivos siguen
   en el almacén.

**Para agentes.** Si buscan los resultados del test, léanlos de r02 (`test_*` y `evaluation/`),
no de `0bb3379f…`. No vuelvan a correr `final.py --split test`. No registren `test_*` en
ninguna otra corrida.
