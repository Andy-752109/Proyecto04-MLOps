# Contrato de evento edge v1

Este contrato permite que captura/inferencia, persistencia, API y portal intercambien
el resultado de una captura sin depender del runtime ni de la técnica de optimización.
La fuente normativa de campos y tipos es
[`contracts/edge-event.v1.schema.json`](../../contracts/edge-event.v1.schema.json).
El validador en `edge/event_validator.py` aplica el schema con verificación de formatos
y añade las comprobaciones semánticas indicadas abajo.

## Flujo de una captura

1. Al capturar, el edge genera un `capture_id` UUID v4 y fija `captured_at` con zona
   horaria explícita. El ID identifica una captura lógica, no un intento de subida.
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
| `capture_id` | UUID v4 de una captura lógica. Se conserva idéntico en todos los reintentos. | Edge al capturar. |
| `captured_at` | Fecha y hora ISO 8601 con zona explícita; `Z` y offsets como `-06:00` son válidos. Una hora sin zona falla. | Edge al capturar. |
| `device_id` | Identificador no vacío del dispositivo. | Configuración local del edge. |
| `model_version` | Versión no vacía del artefacto optimizado usado. | Metadatos del artefacto cargado en edge. |
| `model_sha256` | SHA-256 del mismo artefacto, 64 caracteres hexadecimales. | Metadatos/verificación del artefacto en edge. |
| `runtime` | Cadena no vacía del runtime realmente utilizado; no se limita a ONNX, TFLite ni otra técnica. | Ejecutor de inferencia edge. |
| `predicted_class` | `cat` o `dog`. | Inferencia edge. |
| `confidence` | Número de 0 a 1, inclusive, para la predicción informada. | Inferencia edge. |
| `probabilities` | Objeto con exactamente `cat` y `dog`, ambos números de 0 a 1. Representa las probabilidades producidas para las dos clases. | Inferencia edge. |
| `crop` | `null` si no hay recorte; si lo hay, objeto con `x`, `y`, `width`, `height`, `frame_width`, `frame_height`. Son enteros en píxeles; `x` e `y` son no negativos y las cuatro dimensiones son positivas. | Captura/preprocesamiento edge. |
| `preprocess_ms` | Duración del preprocesamiento en milisegundos, número no negativo. | Edge durante el procesamiento. |
| `inference_ms` | Duración de inferencia en milisegundos, número no negativo. | Edge durante la inferencia. |
| `image_key` | Clave exacta de la fotografía JPG: `edge-captures/v1/images/{capture_id}.jpg`. | Edge al construir el evento. |

El schema no exige que `cat + dog == 1`: la representación de coma flotante vuelve
inapropiada una igualdad exacta. Tampoco se añade una tolerancia semántica. No se
impone por ahora que `confidence` sea igual a la probabilidad de la clase predicha.
Para `crop`, el schema no expresa `x + width <= frame_width` ni
`y + height <= frame_height`; el validador tampoco añade esos límites. Si los
consumidores los necesitan, deben acordarse y probarse como regla semántica aparte.

## Layout, reintentos e idempotencia

| Objeto | Clave S3 |
|---|---|
| Imagen | `edge-captures/v1/images/{capture_id}.jpg` |
| Evento | `edge-captures/v1/events/{capture_id}.json` |

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
`uuid` y `date-time`; además comprueba que `capture_id` sea realmente UUID v4,
que `captured_at` incluya zona y que `image_key` contenga el mismo texto de
`capture_id`. Esto evita que dos representaciones del ID apunten a claves S3
distintas. Se aceptan caracteres hexadecimales en mayúsculas o minúsculas.

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
