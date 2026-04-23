"""Quick smoke test for BGE-Gemma2 embeddings via LiteLLM proxy.

Usage:
    python scripts/test_litellm.py
    python scripts/test_litellm.py --url http://localhost:4000 --model bge-multilingual-gemma2
    python scripts/test_litellm.py --key sk-your-master-key
"""

import argparse
import json
import sys
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description="Test LiteLLM embedding endpoint")
    parser.add_argument("--url", default="http://localhost:4000", help="LiteLLM base URL")
    parser.add_argument("--model", default="bge-multilingual-gemma2", help="Model name registered in LiteLLM")
    parser.add_argument("--key", default="", help="LiteLLM API key (LITELLM_MASTER_KEY)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    texts = [
        "Die Gemeinde bietet Förderungen für Solaranlagen an.",
        "Öffnungszeiten des Gemeindeamts sind Montag bis Freitag.",
    ]

    # ── 1. Health check ──────────────────────────────
    print(f"\n[1/3] Health check: GET {base_url}/health")
    try:
        resp = httpx.get(f"{base_url}/health", timeout=10)
        print(f"      Status: {resp.status_code}")
        if resp.status_code == 200:
            print("      OK: LiteLLM is reachable")
        else:
            print(f"      FAIL: Unexpected status: {resp.text}")
    except httpx.ConnectError:
        print(f"      FAIL: Cannot connect to {base_url}")
        print("        Check that LiteLLM is running and the URL is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"      FAIL: Error: {e}")
        sys.exit(1)

    # ── 2. Model list ────────────────────────────────
    print(f"\n[2/3] Model list: GET {base_url}/v1/models")
    headers = {"Content-Type": "application/json"}
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"

    try:
        resp = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=10)
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            print(f"      Available models: {models}")
            if args.model in models:
                print(f"      OK: '{args.model}' is registered")
            else:
                print(f"      FAIL: '{args.model}' NOT found in model list")
                print(f"        Check your LiteLLM config.yaml model_name")
        else:
            print(f"      FAIL: Status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"      FAIL: Error: {e}")

    # ── 3. Embedding request ─────────────────────────
    print(f"\n[3/3] Embedding: POST {base_url}/v1/embeddings")
    print(f"      Model: {args.model}")
    print(f"      Texts: {len(texts)} samples")

    payload = {"input": texts, "model": args.model}
    start = time.monotonic()

    try:
        resp = httpx.post(
            f"{base_url}/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=120,
        )
    except httpx.ConnectError:
        print(f"      FAIL: Connection refused")
        sys.exit(1)
    except httpx.ReadTimeout:
        print(f"      FAIL: Request timed out (120s) — model may be loading")
        sys.exit(1)
    except Exception as e:
        print(f"      FAIL: Error: {e}")
        sys.exit(1)

    duration_ms = int((time.monotonic() - start) * 1000)

    if resp.status_code != 200:
        print(f"      FAIL: HTTP {resp.status_code}")
        print(f"      Response: {resp.text[:500]}")
        sys.exit(1)

    data = resp.json()
    embeddings = sorted(data.get("data", []), key=lambda x: x["index"])

    if not embeddings:
        print("      FAIL: No embeddings returned")
        sys.exit(1)

    dims = [len(e["embedding"]) for e in embeddings]
    print(f"      OK: Success in {duration_ms}ms")
    print(f"      Embeddings returned: {len(embeddings)}")
    print(f"      Dimensions: {dims}")
    print(f"      First 5 values: {embeddings[0]['embedding'][:5]}")

    usage = data.get("usage", {})
    if usage:
        print(f"      Tokens: {json.dumps(usage)}")

    # Sanity checks
    print("\n-- Summary --")
    if all(d == dims[0] for d in dims):
        print(f"  OK: All embeddings have consistent dimension: {dims[0]}")
    else:
        print(f"  FAIL: Inconsistent dimensions: {dims}")

    if dims[0] == 3584:
        print(f"  OK: Dimension matches BGE_GEMMA2_DENSE_DIM default (3584)")
    else:
        print(f"  WARN: Dimension is {dims[0]}, update BGE_GEMMA2_DENSE_DIM in .env accordingly")

    print()


if __name__ == "__main__":
    main()
