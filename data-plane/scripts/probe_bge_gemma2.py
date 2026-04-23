"""Probe the configured LiteLLM endpoint to see if it actually returns
a BGE-Gemma2 embedding vector for our input.

Replicates BGEGemma2Client.embed_batch() exactly so whatever breaks here
is exactly what breaks during ingest.

Run:
    .venv/Scripts/python.exe scripts/probe_bge_gemma2.py

Optional input override:
    .venv/Scripts/python.exe scripts/probe_bge_gemma2.py "some text to embed"
"""

import asyncio
import json
import sys

import httpx

from app.config import ext


async def main() -> None:
    base_url = ext.litellm_url.rstrip("/")
    model = ext.bge_gemma2_model
    api_key = ext.litellm_api_key
    expected_dim = ext.bge_gemma2_dense_dim

    texts = sys.argv[1:] or ["Förderungen der Gemeinde Wiener Neudorf – Testtext."]

    print("── LiteLLM config ─────────────────────────────")
    print(f"  URL:          {base_url}/v1/embeddings")
    print(f"  Model:        {model}")
    print(f"  API key set:  {bool(api_key)}")
    print(f"  Expected dim: {expected_dim}")
    print(f"  Inputs:       {len(texts)}")
    print()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
        try:
            resp = await client.post(
                f"{base_url}/v1/embeddings",
                headers=headers,
                json={"input": texts, "model": model},
            )
        except httpx.RequestError as e:
            print(f"✗ CONNECTION ERROR: {e}")
            print("  → LiteLLM is unreachable. Check DP env `litellm_url` and that the proxy is running.")
            return

    print(f"── Response ────────────────────────────────────")
    print(f"  Status: {resp.status_code}")
    print(f"  Content-Type: {resp.headers.get('content-type')}")

    try:
        body = resp.json()
    except Exception:
        print("✗ Response body is not JSON:")
        print(resp.text[:500])
        return

    if resp.status_code >= 400:
        print("✗ HTTP error body:")
        print(json.dumps(body, indent=2)[:800])
        return

    data = body.get("data")
    if not isinstance(data, list) or not data:
        print("✗ Response has no 'data' array:")
        print(json.dumps(body, indent=2)[:800])
        return

    print(f"  data[] length: {len(data)}")
    for i, item in enumerate(data):
        if "embedding" not in item:
            print(f"✗ data[{i}] has no 'embedding' key. Keys: {list(item.keys())}")
            continue
        emb = item["embedding"]
        if not isinstance(emb, list) or not emb:
            print(f"✗ data[{i}].embedding is not a non-empty list (got {type(emb).__name__})")
            continue
        all_floats = all(isinstance(x, (int, float)) for x in emb)
        dim = len(emb)
        dim_ok = dim == expected_dim
        print(
            f"  data[{i}]: dim={dim} "
            f"{'✓' if dim_ok else '✗ MISMATCH (expected %d)' % expected_dim} "
            f"all_numeric={'✓' if all_floats else '✗'} "
            f"sample={emb[:3]}…"
        )

    usage = body.get("usage")
    if usage:
        print(f"  usage: {usage}")


if __name__ == "__main__":
    asyncio.run(main())
