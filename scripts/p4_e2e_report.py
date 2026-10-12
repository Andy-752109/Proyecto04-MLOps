"""Genera la evidencia S3, portal y resumen de la validación E2E (P4-12 y P4-16).

Lee `{OUT}/{captures.jsonl,ground_truth.csv}` (copiados de la laptop edge), consulta S3 con
el perfil SSO y la API del portal, y escribe en esa carpeta `events_s3.json` y
`trazabilidad.md`; además `resumen.md` (P4-12) o, con `--solo-trazabilidad` (P4-16),
`s3_consulta.txt` y `portal_items.json` de los 5 IDs, sin URL prefirmadas.

P4-12, con el portal local en :8080:

    AWS_PROFILE=mlops-p3 python3 scripts/p4_e2e_report.py ID1 ID2 ID3 ID4 ID5

P4-16, con el portal desplegado en AWS. El basic auth se toma de PORTAL_USER y
PORTAL_PASSWORD; nunca se escribe en los archivos de salida:

    AWS_PROFILE=mlops-p3 PORTAL_USER=evaluador PORTAL_PASSWORD=... python3 scripts/p4_e2e_report.py \
        --out reports/p4/operation/portal-aws --portal http://18.213.105.30 --solo-trazabilidad \
        ID1 ID2 ID3 ID4 ID5
"""

import argparse
import base64
import csv
import json
import os
import subprocess
import urllib.request
from pathlib import Path

BUCKET = "mlops-p4-edge-captures-222629887955"
PREFIX = "edge-captures/v1/"


def aws(*args: str) -> dict:
    out = subprocess.run(["aws", "s3api", *args, "--output", "json"], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def portal_items(base: str) -> dict:
    request = urllib.request.Request(f"{base.rstrip('/')}/ml-api/edge/captures?limit=100")
    user, password = os.environ.get("PORTAL_USER"), os.environ.get("PORTAL_PASSWORD")
    if user and password:
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    return {i["capture_id"]: i for i in json.load(urllib.request.urlopen(request, timeout=60))["items"]}


def main(ids5: list[str], out: Path, portal_url: str, solo_trazabilidad: bool) -> None:
    local = {}
    for line in (out / "captures.jsonl").read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)["event"]
        local[ev["capture_id"]] = ev
    truth = {r["capture_id"]: r for r in csv.DictReader((out / "ground_truth.csv").open())}

    listing = aws("list-objects-v2", "--bucket", BUCKET, "--prefix", PREFIX)["Contents"]
    modified = {o["Key"]: o for o in listing}
    s3 = {}
    for cid in ids5 if solo_trazabilidad else local:
        key = f"{PREFIX}events/{cid}.json"
        body = subprocess.run(
            ["aws", "s3", "cp", f"s3://{BUCKET}/{key}", "-"], check=True, capture_output=True, text=True
        ).stdout
        s3[cid] = {"key": key, "last_modified": modified[key]["LastModified"], "event": json.loads(body)}
    (out / "events_s3.json").write_text(json.dumps(list(s3.values()), indent=2), encoding="utf-8")

    portal = portal_items(portal_url)

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
    (out / "trazabilidad.md").write_text(
        "# Trazabilidad de 5 capture_id\n\n"
        f"Portal consultado: `{portal_url}` (`GET /ml-api/edge/captures?limit=100`).\n\n"
        "| `capture_id` | Log local | S3 | Portal (`GET /edge/captures`) |\n|---|---|---|---|\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    if solo_trazabilidad:
        consulta = [f"aws s3api list-objects-v2 --bucket {BUCKET} --prefix {PREFIX}  (filtrado a los 5 IDs)", ""]
        for cid in ids5:
            for key in (f"{PREFIX}events/{cid}.json", f"{PREFIX}images/{cid}.jpg"):
                o = modified.get(key)
                consulta.append(f"{o['LastModified']}  {o['Size']:>7}  {key}" if o else f"(no existe)  {key}")
        (out / "s3_consulta.txt").write_text("\n".join(consulta) + "\n", encoding="utf-8")
        # Sin `image_url`: es una URL prefirmada y no va en Git.
        sin_url = [{k: v for k, v in portal[c].items() if k != "image_url"} for c in ids5 if c in portal]
        (out / "portal_items.json").write_text(json.dumps(sin_url, indent=2), encoding="utf-8")
        print(*rows, sep="\n")
        return

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
    (out / "resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(*lines, sep="\n")
    print(*rows, sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs=5, metavar="ID")
    parser.add_argument("--out", type=Path, default=Path("reports/p4/operation"))
    parser.add_argument("--portal", default="http://localhost:8080", help="URL base del portal")
    parser.add_argument("--solo-trazabilidad", action="store_true", help="P4-16: sin resumen.md de P4-12")
    args = parser.parse_args()
    main(args.ids, args.out, args.portal, args.solo_trazabilidad)
