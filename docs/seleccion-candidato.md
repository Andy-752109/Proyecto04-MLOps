# Selección del candidato (P3-11 / #18)

Selección **por validación**, sellada antes de tocar el test. Métrica de la
sección 5 de `docs/decisiones-proyecto3.md`: `best_val_accuracy` (mayor),
desempate por `best_val_macro_f1` (mayor) y luego menor `best_val_loss`.

- **Ganadora:** `r02` · `run_id` `7e7b4a4b35464cfebb6b41714a3ad931`
- **Manifiesto:** `v0.1.1-53fc84fdaa07` · `manifest_sha256` `53fc84fdaa0715f37d90324ae49be39034e2a67bc09ae0a1f1fec8425aeec80f`
- **`checkpoint_sha256`:** `f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9`
- **`test_ids_sha256` (sello del test):** `2ff781befc9d118ead9f9af49274fa541e63457d89be306033d22bd563eb4b22`

El candado (`app/selection/lock.py`) exige que ese `test_ids_sha256` coincida
con el manifiesto para dejar correr la evaluación en `test`; la API de
Evaluation se niega hasta que exista `reports/selection.json`. Ese archivo —y el
`selected_at`— los produce `app/select_candidate.py` cuando Karen declara
**MODEL SELECTION CLOSED** (todavía no se ha sellado).

## Ranking (por la métrica de selección)

| # | Corrida | `run_id` | `best_val_accuracy` | `best_val_macro_f1` | `best_val_loss` |
|---|---|---|---|---|---|
| 1 | r02 ✅ | `7e7b4a4b35464cfebb6b41714a3ad931` | 0.9618 | 0.9618 | 0.1233 |
| 2 | r06 | `0d202fdb6fd849a59a1423eaea882165` | 0.9542 | 0.9542 | 0.2127 |
| 3 | r03 | `0430937ba09d49a78ae67abd62da26cb` | 0.9466 | 0.9465 | 0.0838 |
| 4 | r08 | `e91087b8d30a43abb91f159acecfd886` | 0.9466 | 0.9465 | 0.0923 |
| 5 | r09 | `00143fe56a564f9a8226c67004c13bdc` | 0.9466 | 0.9465 | 0.1254 |
| 6 | r05 | `9f0caa6305f1477795596e930ab7a28b` | 0.9389 | 0.9389 | 0.1100 |
| 7 | r07 | `e7bf686b020f4d02a3fc3e1462e9da33` | 0.9313 | 0.9313 | 0.1077 |
| 8 | r01 | `5e9ae5ebd2e146528a15a4cb5286795d` | 0.9313 | 0.9313 | 0.1571 |
| 9 | r04 | `f13d017045c24e8aa59320e6cbcb9126` | 0.9313 | 0.9312 | 0.1130 |
| 10 | r10 | `7fe4e6afbe424331934017a11a0b2f17` | 0.9160 | 0.9160 | 0.1650 |

## Ganadora: `r02`

Mayor `best_val_accuracy` de la rejilla (0.9618) y,
entre las de ese valor, el mayor `best_val_macro_f1` (0.9618).
adam, batch 32, 15 épocas máx., lr 3e-4, imagen 128, 1 capa oculta, dropout 0.3.
Al cerrarse la selección, `app/select_candidate.py` etiqueta esta corrida en
MLflow con `selected_candidate`, `selected_at` y `selection_metric`.

## Descartadas (2)

La regla es solo la métrica de la sección 5 (`best_val_accuracy`; desempate por
`best_val_macro_f1`, luego menor `best_val_loss`):

- **`r06`** (`0d202fdb6fd849a59a1423eaea882165`): `best_val_accuracy` 0.9542,
  menor que el 0.9618 de r02.
- **`r04`** (`f13d017045c24e8aa59320e6cbcb9126`): `best_val_accuracy` 0.9313;
  en el empate de 0.9313 con r07/r01, su `best_val_macro_f1` (0.9312) es menor
  que el de r07 (0.9313) y r01 (0.9313).

Con la selección cerrada, el test queda bajo custodia de Karen: `final.py`
(`--split test`, P3-13) solo corre después de que ella declare **MODEL SELECTION CLOSED**.
