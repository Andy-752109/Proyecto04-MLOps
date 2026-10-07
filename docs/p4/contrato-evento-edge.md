# Contrato de evento edge v1

Este contrato permite que captura/inferencia, persistencia, API y portal intercambien
el resultado de una captura sin depender del runtime ni de la técnica de optimización.
La fuente normativa de campos y tipos es
[`contracts/edge-event.v1.schema.json`](../../contracts/edge-event.v1.schema.json).
El validador en `edge/event_validator.py` aplica el schema con verificación de formatos
y añade las comprobaciones semánticas indicadas abajo.

## Flujo de una captura

1. Al capturar el frame, el edge genera una sola vez un `capture_id` UUID v4
   canónico lowercase y fija `captured_at` con zona horaria explícita. El ID
   identifica una captura lógica, no un intento de subida.
2. El edge produce la fotografía JPG y, si corresponde, el crop; preprocesa y ejecuta
   inferencia local. Anota versión y SHA-256 del artefacto optimizado efectivamente usado.
3. El edge construye y valida el evento. `image_key` usa el mismo `capture_id`.
4. El futuro uploader sube **primero la imagen** y **después el evento**. La existencia
   del objeto de evento indica que la captura está completa.

Este PR solo define y valida el evento; no implementa captura, inferencia ni subida.

## Campos obligatorios

El evento contiene exactamente los siguientes campos. Los nombres no listados se
rechazan, también dentro de `probabilities` y del objeto `crop`.

| Campo | Significado y restricciones | Lo genera |
|---|---|---|
| `schema_version` | Cadena literal `"1"`. | Código edge al construir el evento. |
| `capture_id` | UUID v4 canónico en minúsculas de una captura lógica. Se conserva idéntico en todos los reintentos. | Edge al capturar. |
| `captured_at` | Fecha y hora ISO 8601 válida con zona explícita; `Z` y offsets como `-06:00` son válidos. Una hora inválida o sin zona falla. | Edge al capturar. |
| `device_id` | Identificador no vacío del dispositivo. | Configuración local del edge. |
| `model_version` | Versión no vacía del artefacto optimizado usado. | Metadatos del artefacto cargado en edge. |
| `model_sha256` | SHA-256 del mismo artefacto, 64 caracteres hexadecimales. | Metadatos/verificación del artefacto en edge. |
| `runtime` | Cadena no vacía del runtime realmente utilizado; no se limita a ONNX, TFLite ni otra técnica. | Ejecutor de inferencia edge. |
| `predicted_class` | `cat` o `dog`. | Inferencia edge. |
| `confidence` | Número finito de 0 a 1, inclusive, para la predicción informada. | Inferencia edge. |
| `probabilities` | Objeto con exactamente `cat` y `dog`, ambos números finitos de 0 a 1. Representa las probabilidades producidas para las dos clases. | Inferencia edge. |
| `crop` | `null` si no hay recorte; si lo hay, objeto con `x`, `y`, `width`, `height`, `frame_width`, `frame_height`. Son enteros en píxeles del frame original; `x` e `y` son no negativos y las cuatro dimensiones son positivas. | Captura/preprocesamiento edge. |
| `preprocess_ms` | Duración finita y no negativa desde el frame o crop hasta obtener el tensor de entrada; incluye resize y normalización. | Edge durante el procesamiento. |
| `inference_ms` | Duración finita y no negativa de la llamada al runtime de inferencia; excluye carga del modelo y warm-up. | Edge durante la inferencia. |
| `image_key` | Clave exacta de la fotografía JPG: `edge-captures/v1/images/{capture_id}.jpg`, con el mismo UUID canónico lowercase. | Edge al construir el evento. |

El schema no exige que `cat + dog == 1`: la representación de coma flotante vuelve
inapropiada una igualdad exacta. Sí se exige consistencia de la predicción (P4-06):
`predicted_class` es la clase de mayor probabilidad (en un empate vale cualquiera) y
`confidence` es igual a `probabilities[predicted_class]` con tolerancia 1e-6, porque
el productor los genera de la misma salida del modelo.
Las coordenadas de `crop` usan píxeles del frame original, antes del resize, con
origen en la esquina superior izquierda. `image_key` apunta al frame completo;
`crop` identifica la región de ese frame utilizada para inferencia. Así el portal
podrá representar esa región más adelante. Cuando `crop` no es `null`, el
validador exige semánticamente `x + width <= frame_width` y
`y + height <= frame_height`; estas relaciones no están en JSON Schema.

`captured_at` marca la captura del frame y representa el reloj de pared del
dispositivo. El productor utilizará `datetime.now(timezone.utc)`. Puede haber
desviación si el dispositivo estuvo offline; revisar y sincronizar ese reloj
corresponde a P4-12. `preprocess_ms` e `inference_ms` son duraciones, no timestamps:
el productor las medirá con `time.perf_counter()`. Ambas deben ser finitas y >= 0.

## Layout, reintentos e idempotencia

| Objeto | Clave S3 |
|---|---|
| Imagen | `edge-captures/v1/images/{capture_id}.jpg` |
| Evento | `edge-captures/v1/events/{capture_id}.json` |

El `capture_id` se conserva localmente junto con el evento. Todos los reintentos
reutilizan exactamente ese ID: nunca se genera uno nuevo por intento de upload.
El futuro uploader usa el orden **imagen → evento** y solicitudes condicionales
`If-None-Match: *`. Una respuesta HTTP `412` significa que el objeto ya existe y
se considera ya enviado. Ante un reintento se reutilizan el mismo `capture_id`,
las mismas claves y el mismo contenido de esa captura; no se genera otra inferencia
para ese intento. La clave del evento es la clave de idempotencia y su existencia
señala la captura completa. Esta es solo la semántica acordada; no hay código S3
en este PR.

`received_at` **no** viaja en el evento. El lector lo deriva después de
`LastModified` del objeto de evento en S3. No debe confundirse con `captured_at`.
El tiempo de subida (`upload_ms`) pertenece solo al log local. También quedan fuera
del evento las access keys, tokens, contraseñas, endpoints y detalles del uploader.

## Validación

JSON Schema Draft 2020-12 valida tipos, campos obligatorios, rangos, enumeración,
formas estrictas, patrones y formatos. El validador activa `FormatChecker` para
`uuid` y `date-time` (habilitado por `rfc3339-validator`); además comprueba que
`capture_id` sea UUID v4 canónico lowercase, que `captured_at` sea parseable e
incluya zona, que `image_key` contenga el mismo ID y que todos los números sean
finitos. Esto evita que dos representaciones del ID apunten a claves S3 distintas.
El CI ejecuta los tests offline de `edge/tests` en un job propio.

## Ejemplo completo

```json
{
  "schema_version": "1",
  "capture_id": "8f9f60e4-6d7a-4d92-8cc0-45e0b192cd65",
  "captured_at": "2026-10-05T12:34:56-06:00",
  "device_id": "edge-demo-01",
  "model_version": "optimized-v1",
  "model_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "runtime": "local-runtime",
  "predicted_class": "cat",
  "confidence": 0.93,
  "probabilities": { "cat": 0.93, "dog": 0.07 },
  "crop": null,
  "preprocess_ms": 12.5,
  "inference_ms": 38.2,
  "image_key": "edge-captures/v1/images/8f9f60e4-6d7a-4d92-8cc0-45e0b192cd65.jpg"
}
```

El mismo JSON está en `contracts/examples/valid-cat-no-crop.json`. La fotografía
`contracts/examples/test-capture.jpg` es un fixture sintético pequeño, sin datos
reales; las claves S3 del ejemplo son convenciones, no objetos subidos.
