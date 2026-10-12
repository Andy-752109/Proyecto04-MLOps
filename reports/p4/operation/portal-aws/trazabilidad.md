# Trazabilidad de 5 capture_id

Portal consultado: `http://18.213.105.30` (`GET /ml-api/edge/captures?limit=100`).

| `capture_id` | Log local | S3 | Portal (`GET /edge/captures`) |
|---|---|---|---|
| `d9bd109d-784e-426f-acb5-8743e44fef1e` | 2026-10-09T00:51:01.835899+00:00 · cat 0.778 (real: dog) | 2026-10-09T00:51:06+00:00 · 2 objetos · igual al local | en el portal, metadatos iguales |
| `3e60aa20-2b0e-4ed7-bf02-2ff3f2d6f02e` | 2026-10-09T00:53:06.999265+00:00 · dog 0.995 (real: dog) | 2026-10-09T00:53:12+00:00 · 2 objetos · igual al local | en el portal, metadatos iguales |
| `45730efb-5920-4eb1-9874-c41369252764` | 2026-10-09T00:58:18.782658+00:00 · cat 0.998 (real: cat) | 2026-10-09T00:58:23+00:00 · 2 objetos · igual al local | en el portal, metadatos iguales |
| `015513c6-2ef3-4000-9e1a-45cdf4358fd5` | 2026-10-11T23:39:38.903572+00:00 · dog 0.999 (real: dog) | 2026-10-11T23:39:37+00:00 · 2 objetos · igual al local | en el portal, metadatos iguales |
| `0ee4aa1b-52d1-49db-8265-e042d954c4e5` | 2026-10-12T01:38:06.100136+00:00 · cat 0.999 (real: cat) | 2026-10-12T01:38:16+00:00 · 2 objetos · igual al local | en el portal, metadatos iguales |
