# Evidencia de mutaciones (P3-17, #27)

Generado: 2026-10-03T05:25:45.956247+00:00

Cada mutación corre en un `git worktree` temporal (nunca se toca el checkout real) y debe hacer fallar la prueba que la cubre.

| Mutación | Archivo | Prueba | Resultado |
|---|---|---|---|
| grupo duplicado en train y test | `manifest/check_leakage.py` | `tests/test_manifest.py` | matada |
| predicción cambiada en la matriz de confusión | `evaluation/metrics.py` | `tests/test_p3_13.py` | matada |
| aumentación en validación | `training/preprocess.py` | `tests/test_p3_05.py` | matada |
| restaurar la última época en vez de la mejor | `training/trainer.py` | `tests/test_p3_08.py` | matada |
| caja degenerada | `analyzers/invalid_boxes.py` | `tests/test_invalid_boxes.py` | matada |

## grupo duplicado en train y test

`find_leakage` ya no detecta un `duplicate_group_id` (ni `crop_id`/`source_image_id`) que aparece en dos particiones -- sección 8: una fuga real pasaría sin que nadie se entere.

- Archivo: `manifest/check_leakage.py`
- Prueba: `tests/test_manifest.py`
- Resultado: matada

```
bb5e210>

    def test_check_leakage_cli_exit_codes(result, tmp_path: Path, capsys):
        (tmp_path / "v0.1.1").mkdir()
        (tmp_path / "v0.1.1" / "manifest.csv").write_bytes(manifest_csv_bytes(result.rows))
        assert check_leakage_main(["--release", "v0.1.1"], manifests_dir=tmp_path) == 0
        assert json.loads(capsys.readouterr().out)["leakage"] is False
    
        rows = [dict(row) for row in result.rows]
        val = next(r for r in rows if r["split"] == "val")
        next(r for r in rows if r["split"] == "train")["source_image_id"] = val["source_image_id"]
        (tmp_path / "v9.9.9").mkdir()
        (tmp_path / "v9.9.9" / "manifest.csv").write_bytes(manifest_csv_bytes(rows))
>       assert check_leakage_main(["--release", "v9.9.9"], manifests_dir=tmp_path) == 1
E       AssertionError: assert 0 == 1
E        +  where 0 = check_leakage_main(['--release', 'v9.9.9'], manifests_dir=PosixPath('/private/var/folders/lt/f9vxl_bs02n9lhf1xnjb27gc0000gn/T/pytest-of-jpdelmuro/pytest-142/test_check_leakage_cli_exit_co0'))

tests/test_manifest.py:269: AssertionError
----------------------------- Captured stdout call -----------------------------
{
  "manifest": "/private/var/folders/lt/f9vxl_bs02n9lhf1xnjb27gc0000gn/T/pytest-of-jpdelmuro/pytest-142/test_check_leakage_cli_exit_co0/v9.9.9/manifest.csv",
  "rows": 138,
  "leakage": false,
  "intersections": {
    "crop_id": 0,
    "source_image_id": 0,
    "duplicate_group_id": 0
  },
  "details": {
    "crop_id": [],
    "source_image_id": [],
    "duplicate_group_id": []
  }
}
=========================== short test summary info ============================
FAILED tests/test_manifest.py::test_find_leakage_detects_each_kind[crop_id]
FAILED tests/test_manifest.py::test_find_leakage_detects_each_kind[source_image_id]
FAILED tests/test_manifest.py::test_find_leakage_detects_each_kind[duplicate_group_id]
FAILED tests/test_manifest.py::test_check_leakage_cli_exit_codes - AssertionE...
4 failed, 22 passed in 0.81s
```

## predicción cambiada en la matriz de confusión

`confusion_matrix` acumula en `[predicted][true]` en vez de `[true][predicted]` -- la matriz queda transpuesta; precisión y recall se intercambian sin que ninguna métrica agregada avise.

- Archivo: `evaluation/metrics.py`
- Prueba: `tests/test_p3_13.py`
- Resultado: matada

```
-07

tests/test_p3_13.py:104: AssertionError
___________________ test_recompute_matches_the_pure_metrics ____________________

    def test_recompute_matches_the_pure_metrics():
        predictions = fixture_predictions()
        metrics = evaluate(predictions, CLASSES)
        recomputed = sklearn_metrics(
            [p.true_label for p in predictions], [p.predicted_label for p in predictions]
        )
        assert recomputed["accuracy"] == pytest.approx(metrics.accuracy)
        assert recomputed["macro_f1"] == pytest.approx(metrics.macro_f1)
>       assert recomputed["confusion_matrix"] == metrics.confusion_matrix
E       assert [[3, 1], [2, 4]] == [[3, 2], [1, 4]]
E         
E         At index 0 diff: [3, 1] != [3, 2]
E         Use -v to get more diff

tests/test_p3_13.py:132: AssertionError
=============================== warnings summary ===============================
tests/test_p3_13.py::test_final_test_refuses_when_the_candidate_already_has_test_metrics
  /Users/jpdelmuro/Documents/Tareas/Semestre 9/Integracion/Proyecto 3/app/.venv/lib/python3.12/site-packages/mlflow/store/tracking/utils/sql_trace_metrics_utils.py:155: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1) (This warning originated from the `configure_mappers()` process, which was invoked automatically in response to a user-initiated operation.)
    _SESSION_TRACE_METADATA = aliased(SqlTraceMetadata)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_p3_13.py::test_evaluate_hand_computed_metrics - assert [[3,...
FAILED tests/test_p3_13.py::test_analysis_reports_baseline_most_confused_and_recall
FAILED tests/test_p3_13.py::test_recompute_matches_the_pure_metrics - assert ...
3 failed, 10 passed, 1 warning in 40.97s
```

## aumentación en validación

`get_preprocessing_transforms` aplica el pipeline de 'train' (recorte aleatorio, flip, color jitter) a cualquier split que no sea 'test' -- validation deja de ser determinista.

- Archivo: `training/preprocess.py`
- Prueba: `tests/test_p3_05.py`
- Resultado: matada

```
.....F                                                                   [100%]
=================================== FAILURES ===================================
_______________________ test_preprocessing_deterministic _______________________

    def test_preprocessing_deterministic():
        transform = get_preprocessing_transforms(split="val", image_size=128)
        img_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
    
        tensor1 = transform(img)
        tensor2 = transform(img)
    
>       assert torch.equal(tensor1, tensor2)
E       assert False
E        +  where False = <built-in method equal of type object at 0x10a4993a0>(tensor([[[-0.4911,  0.2282, -0.1828,  ...,  0.1939, -0.2856, -0.0629],\n         [ 0.2453, -0.1486,  0.4679,  ...,  0.0...58, -0.1835,  ..., -0.3404, -0.6018,  0.7576],\n         [ 0.2173,  0.4614,  0.6705,  ...,  0.0431, -0.2358,  0.6182]]]), tensor([[[-0.2856, -1.9980, -1.6042,  ...,  0.0912,  1.2214,  0.0912],\n         [ 0.5193, -1.3302, -0.0972,  ..., -1.0...82,  1.6465,  ..., -1.4733, -0.4450, -0.9678],\n         [ 1.2457,  0.2522, -0.2532,  ..., -1.0201, -0.8981, -0.0964]]]))
E        +    where <built-in method equal of type object at 0x10a4993a0> = torch.equal

tests/test_p3_05.py:88: AssertionError
=========================== short test summary info ============================
FAILED tests/test_p3_05.py::test_preprocessing_deterministic - assert False
1 failed, 5 passed in 1.05s
```

## restaurar la última época en vez de la mejor

`EarlyStopping` guarda los pesos en cada llamada, no solo cuando `val_loss` mejora -- al terminar, `best_weights` es en realidad la última época, no la de mejor validación.

- Archivo: `training/trainer.py`
- Prueba: `tests/test_p3_08.py`
- Resultado: matada

```

    def __call__(
        self, val_loss: float, val_acc: float, val_f1: float, model: torch.nn.Module, epoch: int
    ):
        if True:  # val_loss < self.best_loss - self.min_delta
            self.best_loss = val_loss
            self.best_val_accuracy = val_acc
            self.best_val_macro_f1 = val_f1
>           self.best_weights = copy.deepcopy(model.state_dict())
                                              ^^^^^^^^^^^^^^^^
E           AttributeError: 'NoneType' object has no attribute 'state_dict'

training/trainer.py:63: AttributeError
----------------------------- Captured stderr call -----------------------------
2026/10/02 23:25:44 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/10/02 23:25:44 INFO mlflow.store.db.utils: Updating database tables
2026/10/02 23:25:44 INFO mlflow.tracking.fluent: Experiment with name 'clasificador-perro-gato' does not exist. Creating a new experiment.
=============================== warnings summary ===============================
tests/test_p3_08.py::test_mlflow_contract
  /Users/jpdelmuro/Documents/Tareas/Semestre 9/Integracion/Proyecto 3/app/.venv/lib/python3.12/site-packages/mlflow/store/tracking/utils/sql_trace_metrics_utils.py:155: SADeprecationWarning: The ``noload`` loader strategy is deprecated and will be removed in a future release.  This option produces incorrect results by returning ``None`` for related items. (deprecated since: 2.1) (This warning originated from the `configure_mappers()` process, which was invoked automatically in response to a user-initiated operation.)
    _SESSION_TRACE_METADATA = aliased(SqlTraceMetadata)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_p3_08.py::test_early_stopping_controlled_sequence - Asserti...
FAILED tests/test_p3_08.py::test_mlflow_fails_on_nan_val_loss - AttributeErro...
2 failed, 12 passed, 1 warning in 34.11s
```

## caja degenerada

`analyze_invalid_boxes` ya no marca una bbox de ancho exactamente cero como inválida (solo ancho negativo) -- una caja degenerada pasaría la compuerta de calidad.

- Archivo: `analyzers/invalid_boxes.py`
- Prueba: `tests/test_invalid_boxes.py`
- Resultado: matada

```
..F...........................                                           [100%]
=================================== FAILURES ===================================
_____________ test_each_geometric_error[bbox2-non_positive_width] ______________

bbox = [0, 0, 0, 10], reason = 'non_positive_width'

    @pytest.mark.parametrize(
        "bbox, reason",
        [
            ([0, 0, -10, 10], "non_positive_width"),
            ([0, 0, 10, -10], "non_positive_height"),
            ([0, 0, 0, 10], "non_positive_width"),
            ([0, 0, 10, 0], "non_positive_height"),
            ([-1, 0, 10, 10], "negative_x"),
            ([0, -1, 10, 10], "negative_y"),
            ([91, 0, 10, 10], "exceeds_image_width"),
            ([0, 71, 10, 10], "exceeds_image_height"),
        ],
    )
    def test_each_geometric_error(bbox, reason):
        result = analyze_invalid_boxes(document(bbox, bbox[2] * bbox[3]), load_invalid_box_config())
>       assert result.metric_value == 1
E       AssertionError: assert 0.0 == 1
E        +  where 0.0 = AnalyzerResult(check_name='degenerate_boxes', passed=True, metric_value=0.0, details={'total_annotations': 1, 'invalid_annotations': 0, 'offending_samples': []}).metric_value

tests/test_invalid_boxes.py:49: AssertionError
=========================== short test summary info ============================
FAILED tests/test_invalid_boxes.py::test_each_geometric_error[bbox2-non_positive_width]
1 failed, 29 passed in 0.09s
```
