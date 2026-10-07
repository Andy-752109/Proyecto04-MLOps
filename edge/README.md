# Edge: contrato de evento v1

Este directorio contiene solo el validador del contrato compartido. Su fuente de verdad
es [`contracts/edge-event.v1.schema.json`](../contracts/edge-event.v1.schema.json); el
flujo y las reglas están en [`docs/p4/contrato-evento-edge.md`](../docs/p4/contrato-evento-edge.md).

Desde la raíz del repositorio, con `jsonschema>=4.23,<5` disponible:

```bash
python3 -m unittest discover -s edge/tests -v
```

`validate_event(event)` recibe un objeto Python que representa el JSON y devuelve
`None` si es válido; de lo contrario lanza `EventValidationError` con el campo y el
motivo. No requiere AWS, cámara, modelo ni configuración YAML. `config.example.yaml`
solo ilustra identificadores locales; este PR no implementa su lectura.

## Uploader a S3 (P4-05)

`edge/uploader.py` sube imagen y luego evento con `If-None-Match: *` y devuelve
`sent`, `already_sent` o `failed`. Detalles, prueba de doble envío y comandos de
lectura en [`docs/p4/aws-capturas.md`](../docs/p4/aws-capturas.md).
