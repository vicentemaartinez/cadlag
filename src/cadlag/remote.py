"""Almacenamiento remoto en Supabase Storage (para la actualización automática).

Dos buckets:
- ``cadlag-state`` (privado): el almacén del motor (data/processed/*.parquet), que cambia en cada
  actualización y por eso no vive en git.
- ``cadlag-data`` (público): lo que lee el sitio (site/public/data/*). Vercel lo descarga al compilar.

Variables de entorno: ``SUPABASE_URL`` y ``SUPABASE_SECRET_KEY`` (la clave secreta / service_role:
solo en GitHub Actions, nunca en el sitio). Opcional: ``VERCEL_DEPLOY_HOOK`` para volver a publicar.
"""
from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path

import requests

from .data import store

STATE_BUCKET = "cadlag-state"
DATA_BUCKET = "cadlag-data"
STATE_FILES = ["matches.parquet", "teams.parquet", "stats.parquet"]
SITE_DIR = store.ROOT / "site" / "public" / "data"


def _cfg():
    url, key = os.environ.get("SUPABASE_URL", "").rstrip("/"), os.environ.get("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        raise SystemExit("Faltan SUPABASE_URL y SUPABASE_SECRET_KEY")
    return url, {"apikey": key, "Authorization": f"Bearer {key}"}


def init() -> None:
    """Crea los dos buckets si no existen."""
    url, h = _cfg()
    for b, public in ((STATE_BUCKET, False), (DATA_BUCKET, True)):
        r = requests.post(f"{url}/storage/v1/bucket", headers=h, json=dict(id=b, name=b, public=public), timeout=30)
        exists = r.status_code == 409 or "already exists" in r.text.lower() or "duplicate" in r.text.lower()
        if r.status_code < 300:
            print(f"bucket {b} ({'público' if public else 'privado'}): creado")
        elif not exists:
            raise SystemExit(f"no se pudo crear el bucket {b}: {r.status_code} {r.text}\n"
                             "Revisa que SUPABASE_URL sea la del proyecto correcto y que la clave sea la secreta (service_role).")


def _upload(bucket: str, name: str, path: Path, h: dict, url: str, cache: str = "60") -> None:
    ctype = mimetypes.guess_type(path.name)[0] or ("application/gzip" if path.suffix == ".gz" else "application/octet-stream")
    if path.suffix == ".json":
        ctype = "application/json"
    r = requests.post(f"{url}/storage/v1/object/{bucket}/{name}", data=path.read_bytes(), timeout=120,
                      headers={**h, "x-upsert": "true", "content-type": ctype, "cache-control": f"max-age={cache}"})
    if r.status_code >= 300:
        raise RuntimeError(f"no se pudo subir {bucket}/{name}: {r.status_code} {r.text}")


def pull_state() -> bool:
    """Descarga el almacén del motor. Si el bucket está vacío (primera vez), deja el del repositorio."""
    init()  # crea los buckets si faltan (no hace nada si ya existen)
    url, h = _cfg()
    got = 0
    for f in STATE_FILES:
        r = requests.get(f"{url}/storage/v1/object/{STATE_BUCKET}/{f}", headers=h, timeout=120)
        if r.status_code == 200:
            (store.PROCESSED / f).write_bytes(r.content)
            got += 1
    print(f"estado: {got} de {len(STATE_FILES)} archivos descargados" + ("" if got else " (se usa el del repositorio)"))
    return got >= 2


def push_state() -> None:
    url, h = _cfg()
    for f in STATE_FILES:
        if (store.PROCESSED / f).exists():
            _upload(STATE_BUCKET, f, store.PROCESSED / f, h, url, cache="0")
    print("estado: subido")


def push_site() -> None:
    url, h = _cfg()
    files = sorted(SITE_DIR.iterdir())
    for p in files:
        _upload(DATA_BUCKET, p.name, p, h, url)
    print(f"sitio: {len(files)} archivos subidos a {DATA_BUCKET}")


def redeploy(wait: int = 65) -> None:
    hook = os.environ.get("VERCEL_DEPLOY_HOOK")
    if not hook:
        print("sin VERCEL_DEPLOY_HOOK: no se vuelve a publicar")
        return
    time.sleep(wait)  # que expire el caché de 60 s de los archivos públicos antes de compilar
    r = requests.post(hook, timeout=30)
    print("Vercel:", r.status_code)
