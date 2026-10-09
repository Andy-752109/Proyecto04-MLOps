"""Genera la evidencia S3, portal y resumen de la validación E2E (P4-12).

Lee `reports/p4/operation/{captures.jsonl,ground_truth.csv}` (copiados de la laptop edge),
consulta S3 con el perfil SSO y la API del portal local, y escribe en esa carpeta:
`events_s3.json`, `trazabilidad.md` y `resumen.md`. Uso, con el portal en :8080:

    AWS_PROFILE=mlops-p3 python3 scripts/p4_e2e_report.py ID1 ID2 ID3 ID4 ID5
"""

import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

OUT = Path("reports/p4/operation")
BUCKET = "mlops-p4-edge-captures-222629887955"
PREFIX = "edge-captures/v1/"
PORTAL = "http://localhost:8080/ml-api/edge/captures?limit=100"


def aws(*args: str) -> dict:
    out = subprocess.run(["aws", "s3api", *args, "--output", "json"], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def main(ids5: list[str]) -> None:
    local = {}
    for line in (OUT / "captures.jsonl").read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)["event"]
        local[ev["capture_id"]] = ev
    truth = {r["capture_id"]: r for r in csv.DictReader((OUT / "ground_truth.csv").open())}

    listing = aws("list-objects-v2", "--bucket", BUCKET, "--prefix", PREFIX)["Contents"]
    modified = {o["Key"]: o for o in listing}
    s3 = {}
    for cid in local:
        key = f"{PREFIX}events/{cid}.json"
        body = subprocess.run(
            ["aws", "s3", "cp", f"s3://{BUCKET}/{key}", "-"], check=True, capture_output=True, text=True
        ).stdout
        s3[cid] = {"key": key, "last_modified": modified[key]["LastModified"], "event": json.loads(body)}
    (OUT / "events_s3.json").write_text(json.dumps(list(s3.values()), indent=2), encoding="utf-8")

    portal = {i["capture_id"]: i for i in json.load(urllib.request.urlopen(PORTAL, timeout=60))["items"]}

    rows = []
    for cid in ids5:
        img = f"{PREFIX}images/{cid}.jpg"
        n_obj = sum(1 for k in (f"{PREFIX}events/{cid}.json", img) if k in modified)
        same_s3 = s3[cid]["event"] == local[cid]
        p = portal.get(cid)
        same_portal = bool(p) and all(p.get(k) == v for k, v in local[cid].items())
        ev = local[cid]
        rows.append(
            f"| `{cid}` | {ev['captured_at']} · {ev['predicted_class']} {ev['confidence']:.3f} "
            f"(real: {truth[cid]['true_class']}) | {s3[cid]['last_modified']} · {n_obj} objetos · "
            f"{'igual al local' if same_s3 else 'DISTINTO'} | "
            f"{'en el portal, metadatos iguales' if same_portal else 'NO coincide / no aparece'} |"
        )
    (OUT / "trazabilidad.md").write_text(
        "# Trazabilidad de 5 capture_id\n\n"
        "| `capture_id` | Log local | S3 | Portal (`GET /edge/captures`) |\n|---|---|---|---|\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    rated = [r for r in truth.values() if r["foto"].isdigit()]
    wrong = [r for r in rated if local[r["capture_id"]]["predicted_class"] != r["true_class"]]
    in_portal = sum(
        1 for cid, ev in local.items() if cid in portal and all(portal[cid].get(k) == v for k, v in ev.items())
    )
    lines = [
        "# Resumen de la corrida E2E",
        "",
        f"- Capturas de la corrida en el log local: {len(local)}; en S3 con evento igual al local: "
        f"{sum(1 for c in local if s3[c]['event'] == local[c])}; en el portal con metadatos iguales: {in_portal}",
        f"- Paso 2 (≥20 capturas con fotos del PM): {len(rated)} capturas, {len(rated) - len(wrong)} aciertos, "
        f"{len(wrong)} errores ({(len(rated) - len(wrong)) / len(rated):.0%}); sin umbral de accuracy",
        "",
        "| foto | real | predicho | confianza | `capture_id` |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(wrong, key=lambda r: int(r["foto"])):
        ev = local[r["capture_id"]]
        lines.append(f"| {r['foto']} | {r['true_class']} | {ev['predicted_class']} | {ev['confidence']:.3f} | `{r['capture_id']}` |")
    (OUT / "resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(*lines, sep="\n")
    print(*rows, sep="\n")


if __name__ == "__main__":
    main(sys.argv[1:])
