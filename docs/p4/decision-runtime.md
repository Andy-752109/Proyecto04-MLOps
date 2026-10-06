# Decisión de runtime y técnica de optimización (P4-03)

Aprobada por el PM el lunes 5 de octubre de 2026 (detalle en el issue #3). Los valores que dependían
de la laptop se confirmaron en la primera prueba, el martes 6.

## 1. Decisión

| Aspecto | Decisión |
|---|---|
| Runtime | ONNX Runtime en CPU. No se usa la GPU (GTX 1050): exigiría drivers/CUDA y no aporta a la optimización para edge. |
| Técnica | INT8 estática (QDQ, per-channel). |
| Versión del modelo | `1.0.0-int8` (ruta `models/edge/1.0.0-int8/`). |
| Calibración | 100 recortes **solo de train**, semilla 42; IDs en `calibration_ids.csv`. |
| Plan B | Caída de accuracy en validation > 2 pp → ajustar calibración (más muestras); si no alcanza → FP16 (`1.0.0-fp16`). |
| Si INT8 no gana latencia | Se mantiene INT8: la reducción de tamaño cuenta como mejora medible y se documenta. |

## 2. Dispositivo edge

| Dato | Valor |
|---|---|
| Equipo / `device_id` | ASUS GL553VD, **`edge-laptop-01`** |
| CPU | Intel Core i5-7300HQ @ 2.50 GHz, 4 núcleos, AVX2 sí; AVX-512 y VNNI no |
| RAM | 8 GB |
| SO | Windows 10 Home Single Language 64-bit (build 19045) |
| Python / Git / AWS CLI | 3.12.10 (64-bit) / 2.55.0 / v2.37.9 |
| Cámara | `GENERAL WEBCAM` USB (UVC, VID 0x1B3F / PID 0x2002), 640×360 con `CAP_DSHOW`. La webcam integrada es `USB2.0 UVC HD Webcam` (VID 0x13D3), 640×480. **El índice de OpenCV no es estable** (la USB fue `1` y luego `0`): se elige por nombre |
| Cortar la red | Apagar el WiFi desde Windows |

Como la CPU tiene AVX2 sin VNNI, la cuantización usa `reduce_range=True` para evitar la saturación de
la instrucción U8S8; la ganancia de latencia es moderada y la de tamaño (~4×) es segura.

## 3. Verificado en la laptop

- ONNX Runtime 1.30.0 (CPU) carga e infiere. **Requiere el Microsoft Visual C++ Redistributable x64**;
  sin él falla con `DLL load failed while importing onnxruntime_pybind11_state`. Hay que instalarlo
  tras cualquier reset de la laptop.
- PyTorch `2.14.0+cpu` y torchvision `0.29.0+cpu` se instalan e importan (`cuda.is_available() = False`),
  así que el modelo original se puede medir en la misma laptop para el benchmark.
- AWS SSO con el perfil `mlops-p3`: la variante `1.0.0-int8` se descarga de S3 con el SHA-256 correcto y carga.
- Latencia indicativa (entrada aleatoria, 10 warmups + 100 mediciones): FP32 5.24 ms, INT8 2.83 ms por
  inferencia. No sustituye el benchmark formal de P4-11.

## 4. Notas para la app edge (P4-06)

- Elegir la cámara por nombre (`camera.name: GENERAL WEBCAM`, ver P4-06), no por índice. `python -m edge cameras` lista los dispositivos. Resolución nativa de la cámara.
- Descartar los primeros 5–10 cuadros (autoexposición).
- Recorte: cuadrado centrado del 80 % del lado menor del frame (configurable).
- Modos: manual (Enter) para pruebas; intervalo de 10 s para la corrida de 5 minutos.
- Se sube el frame completo con las coordenadas del recorte (contrato v1).

## 5. Benchmark (P4-11)

Original (PyTorch) e INT8 en la misma laptop, conectada a la corriente y sin otros programas abiertos;
10 warmups + 100 mediciones.
