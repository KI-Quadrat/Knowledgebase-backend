"""End-to-end smoke check that the BGE-Gemma2 fallback embedding is actually
generated and stored in Qdrant when /api/v1/online/ingest is called with
``vector_config.enable_fallback: true``.

Unlike ``probe_bge_gemma2.py`` (which only hits LiteLLM) and
``test_litellm.py`` (proxy smoke test), this drives the **real ingest endpoint**
and then reads back the upserted points from Qdrant to confirm both
``dense_openai`` and ``dense_bge_gemma2`` vectors are present with the
expected dimensions.

Run:
    .venv/Scripts/python.exe scripts/check_online_ingest_fallback.py
    .venv/Scripts/python.exe scripts/check_online_ingest_fallback.py --dp http://localhost:8000 --qdrant http://localhost:6333

Optional:
    --api-key <X-API-Key>      forwarded to the ingest endpoint (only needed when DP_ONLINE_API_KEYS is set)
    --qdrant-key <key>         forwarded as api-key header to Qdrant
    --keep                     don't delete the test points after the check
"""

import argparse
import os
import pathlib
import sys
import time
import uuid

# Force our data-plane root ahead of anything a stray .pth file in the venv
# might inject, so ``from app.config import ext`` always resolves to THIS repo.
_DP_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DP_ROOT))
os.chdir(_DP_ROOT)  # so .env in data-plane/ is picked up by pydantic-settings

import httpx

# Windows consoles default to cp1252 which can't encode the status glyphs below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import ext


COLLECTION = "fallback_smoke_check"


def _ingest_payload(source_id: str) -> dict:
    return {
        "collection_name": COLLECTION,
        "content": (
            "Förderungen der Gemeinde Wiener Neudorf\n\n"
            "Die Gemeinde bietet verschiedene Förderungen für Photovoltaik, "
            "Wärmepumpen und thermische Sanierung an. Anträge sind beim "
            "Bürgerservice einzureichen."
        ),
        "content_type": ["funding", "renewable_energy"],
        "language": "de",
        "metadata": {
            "assistant_id": "asst_fallback_check",
            "department": ["Bürgerservice"],
            "municipality_id": "wiener-neudorf",
            "source_type": "web",
            "title": "Fallback smoke check",
        },
        "source_id": source_id,
        "url": f"https://example.invalid/fallback-check/{source_id}",
        "vector_config": {
            "enable_fallback": True,
            "search_mode": "semantic",
            "vector_size": 1536,
        },
    }


def _print_header(title: str) -> None:
    print(f"\n-- {title} ".ljust(60, "-"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp", default="http://localhost:8000", help="Data plane base URL")
    parser.add_argument("--qdrant", default=None, help="Qdrant base URL (defaults to DP env qdrant_url)")
    parser.add_argument("--api-key", default="", help="X-API-Key for /online/ingest")
    parser.add_argument("--qdrant-key", default=None, help="api-key header for Qdrant (defaults to DP env qdrant_api_key)")
    parser.add_argument("--keep", action="store_true", help="Skip cleanup of inserted points")
    args = parser.parse_args()

    dp_url = args.dp.rstrip("/")
    qdrant_url = (args.qdrant or ext.qdrant_url).rstrip("/")
    qdrant_key = ext.qdrant_api_key if args.qdrant_key is None else args.qdrant_key
    expected_fallback_dim = ext.bge_gemma2_dense_dim
    expected_primary_dim = 1536

    source_id = f"fallback_check_{uuid.uuid4().hex[:8]}"

    _print_header("Config")
    print(f"  DP URL:            {dp_url}")
    print(f"  Qdrant URL:        {qdrant_url}")
    print(f"  Collection:        {COLLECTION}")
    print(f"  source_id:         {source_id}")
    print(f"  Primary dim:       {expected_primary_dim}")
    print(f"  Fallback dim:      {expected_fallback_dim}")

    qdrant_headers = {"Content-Type": "application/json"}
    if qdrant_key:
        qdrant_headers["api-key"] = qdrant_key

    ingest_headers = {"Content-Type": "application/json"}
    if args.api_key:
        ingest_headers["X-API-Key"] = args.api_key

    # ── 1. POST /online/ingest ─────────────────────────────────────────
    _print_header("POST /api/v1/online/ingest")
    payload = _ingest_payload(source_id)
    start = time.monotonic()
    try:
        r = httpx.post(
            f"{dp_url}/api/v1/online/ingest",
            headers=ingest_headers,
            json=payload,
            timeout=httpx.Timeout(180),
        )
    except httpx.RequestError as e:
        print(f"✗ CONNECTION ERROR: {e}")
        print(f"  → Is the data plane running at {dp_url}?")
        return 2
    duration_ms = int((time.monotonic() - start) * 1000)

    print(f"  HTTP {r.status_code} in {duration_ms}ms")
    try:
        body = r.json()
    except Exception:
        print("✗ Response is not JSON:")
        print(r.text[:600])
        return 2

    if r.status_code != 200 or not body.get("success"):
        print("✗ Ingest failed:")
        print(f"  error:  {body.get('error')}")
        print(f"  detail: {body.get('detail')}")
        return 2

    data = body.get("data") or {}
    print(f"  chunks_created: {data.get('chunks_created')}")
    print(f"  vectors_stored: {data.get('vectors_stored')}")
    print(f"  collection:     {data.get('collection')}")
    if not data.get("vectors_stored"):
        print("✗ No vectors stored — nothing to verify.")
        return 2

    # ── 2. Scroll points back from Qdrant ──────────────────────────────
    _print_header(f"GET points from Qdrant collection '{COLLECTION}'")
    scroll_body = {
        "filter": {"must": [{"key": "metadata.source_id", "match": {"value": source_id}}]},
        "with_payload": False,
        "with_vector": True,
        "limit": 64,
    }
    try:
        r = httpx.post(
            f"{qdrant_url}/collections/{COLLECTION}/points/scroll",
            headers=qdrant_headers,
            json=scroll_body,
            timeout=30,
        )
    except httpx.RequestError as e:
        print(f"✗ CONNECTION ERROR talking to Qdrant: {e}")
        return 2

    if r.status_code != 200:
        print(f"✗ Qdrant scroll HTTP {r.status_code}: {r.text[:400]}")
        return 2

    points = (r.json().get("result") or {}).get("points") or []
    print(f"  points returned: {len(points)}")
    if not points:
        print("✗ No points found for this source_id. Ingest likely didn't reach Qdrant.")
        return 2

    # ── 3. Verify both vectors per point ───────────────────────────────
    _print_header("Vector inspection")
    failures = 0
    for i, p in enumerate(points):
        vectors = p.get("vector") or {}
        primary = vectors.get("dense_openai")
        fallback = vectors.get("dense_bge_gemma2")
        primary_dim = len(primary) if isinstance(primary, list) else None
        fallback_dim = len(fallback) if isinstance(fallback, list) else None

        primary_ok = primary_dim == expected_primary_dim
        fallback_ok = fallback_dim == expected_fallback_dim
        marker = "✓" if (primary_ok and fallback_ok) else "✗"
        print(
            f"  {marker} point[{i}] dense_openai={primary_dim} "
            f"dense_bge_gemma2={fallback_dim}"
        )
        if not primary_ok or not fallback_ok:
            failures += 1

    # ── 4. Cleanup ─────────────────────────────────────────────────────
    if not args.keep:
        _print_header("Cleanup")
        try:
            r = httpx.post(
                f"{qdrant_url}/collections/{COLLECTION}/points/delete",
                headers=qdrant_headers,
                json={"filter": {"must": [{"key": "metadata.source_id", "match": {"value": source_id}}]}},
                timeout=30,
            )
            print(f"  delete HTTP {r.status_code}")
        except httpx.RequestError as e:
            print(f"  WARN: cleanup failed: {e}")

    # ── 5. Verdict ─────────────────────────────────────────────────────
    _print_header("Verdict")
    if failures:
        print(f"✗ {failures}/{len(points)} point(s) missing or wrong-dimension fallback vector.")
        print("  → BGE-Gemma2 fallback is NOT working end-to-end.")
        return 1
    print(f"✓ All {len(points)} point(s) have both dense_openai ({expected_primary_dim}) "
          f"and dense_bge_gemma2 ({expected_fallback_dim}) vectors.")
    print("  → BGE-Gemma2 fallback is wired up and producing real vectors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
