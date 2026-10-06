# Prueba de la app edge en la laptop (P4-06)

Martes 6 de octubre de 2026, `edge-laptop-01` (ASUS GL553VD), cámara USB `GENERAL WEBCAM`,
modelo `1.0.0-int8` (SHA-256 `dffa9cf2670f0f9bb96139ae08779e2ba3af87691791ad770061e3f666290392`),
ONNX Runtime 1.30.0 en CPU. Rama `p4-06-camara-por-nombre` (app de #6 + cámara por nombre, PR #24).
Imágenes de perros y gatos mostradas en una pantalla frente a la cámara.

| | Manual (#1-7) | Intervalo cada 10 s (#8-37) | Total |
|---|---|---|---|
| Capturas | 7 | 30 | **37** |
| Perros / gatos | 4 / 3 | 16 / 14 | 20 / 17 |
| Aciertos | 7 / 7 | 30 / 30 | **37 / 37** |
| Confianza | 0.92–1.00 | perro 0.95–0.99, gato 0.68–0.86 | |
| Inferencia | ~3–4 ms | mediana ≈ 3.2 ms (pico 31 ms) | |

La corrida por intervalo duró 4 min 54 s (14:30:16 → 14:35:10 en `edge.log`).

## Limitaciones
- Las 37 capturas son repeticiones de **8 imágenes distintas** (el intervalo mostró siempre la
  misma: un pastor alemán ×16 y una gata blanca ×14). Valida el flujo completo, no mide accuracy.
- Sin red: se probó con el WiFi apagado. El `ping -n 1 8.8.8.8` dio `Destination host unreachable`
  desde la propia laptop; su `Received = 1 (0% loss)` es una cuenta engañosa de `ping` en Windows.
  El log no registra el estado de la red.
- En una prueba previa con animales reales en casa y encuadre libre hubo errores (gatos clasificados
  como `dog`) y muchas fotos inválidas; no se conserva esa corrida.

## Origen de las imágenes
Las 8 imágenes (4 de perros y 4 de gatos) se obtuvieron de **Google Imágenes**: se buscó "perro" y
"gato" y se eligieron a mano. Se mostraron en la pantalla de un computador frente a la cámara USB.

**No son del dataset del proyecto.** Se comparó cada imagen (vía las características del modelo de
P3) con las 600 imágenes de `data/raw/images` (`cat.N.jpg`, `dog.N.jpg`) y no hay ninguna
coincidencia: las más parecidas son otros animales, con similitud de 0.73 a 0.89. Es una
comprobación razonable, no absoluta, porque las capturas pasan por una pantalla y una cámara de baja
resolución. Las imágenes originales no se incluyen en este repositorio (solo las capturas).

## Contenido
| Archivo | Qué es |
|---|---|
| `resultados.csv` | Una fila por captura: sesión, clase real, predicción, confianza, resultado |
| `captures.jsonl`, `edge.log` | Salida de la app tal cual (`edge/data/`) |
| `images/`, `crops/` | Frame completo y recorte usado para inferir, por `capture_id` |
| `hoja_1.png`, `hoja_2.png` | Hojas de contacto con la predicción de cada captura |

La clase real se asigna por el orden de las imágenes mostradas (#1-4 y #8-23 perros; #5-7 y #24-37 gatos).
