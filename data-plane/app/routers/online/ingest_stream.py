"""
POST /api/v1/online/ingest/stream — Stream ingest progress as Server-Sent Events.

Same request body and semantics as ``POST /api/v1/online/ingest`` — the only
difference is the response: instead of waiting for the full pipeline and
returning one JSON envelope, this endpoint streams one SSE event per pipeline
phase and emits a final ``completed`` (or ``error``) event.

SSE wire format (text/event-stream) — each event is two lines plus a blank::

    event: progress
    data: {"phase": "chunked", "chunks": 12}

Heartbeat comments (``: keepalive``) are sent every 15 s so that reverse
proxies / load balancers don't idle-close the connection on slow backends.
"""

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.config import ext
from app.models.online.ingest import OnlineIngestRequest
from app.routers._ingest_utils import INGEST_ERROR_CODE_MAP
from app.services.ingest.ingest_service import IngestError
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/online", tags=["Online - Ingestion Pipeline"])


# Phases the pipeline may emit, in natural order. Documented for clients.
PROGRESS_PHASES = (
    "started",            # {source_id}
    "chunked",            # {chunks}
    "enriched",           # {chunks} — only when chunking.strategy = "contextual"
    "embedded",           # {chunks, has_openai, has_bge_gemma2, duration_ms}
    "funding_extracted",  # {fields: [...]} — only when assistant_type = "funding"
    "stored",             # {vectors, collection}
)

HEARTBEAT_INTERVAL_SECONDS = 15


SSE_STREAM_DESCRIPTION = """\
Streams the ingestion pipeline as **Server-Sent Events** so the client can
display real-time progress (chunked → enriched → embedded → funding_extracted →
stored → completed) instead of waiting for one big response.

The request body is identical to `POST /api/v1/online/ingest`. Only the
response shape differs.

---

## Event protocol

Every event is of the form:

```text
event: <name>
data: <JSON>

```

### Event names

| `event:` | When | `data:` payload |
|---|---|---|
| `progress` | Fires once per pipeline phase | `{"phase": "<phase>", ...}` |
| `completed` | Final event on success. Connection closes right after. | `{"source_id", "chunks_created", "vectors_stored", "collection", "content_type", "embedding_time_ms", "total_time_ms"}` |
| `error` | Final event on failure. Connection closes right after. | `{"code": "<ErrorCode>", "detail": "<message>"}` |

### `progress` phases

| `phase` | Extra fields in payload |
|---|---|
| `started` | `source_id` |
| `chunked` | `chunks` |
| `enriched` | `chunks` (may include `error` if contextual enrichment failed — non-fatal). Only emitted when `chunking.strategy = "contextual"`. |
| `embedded` | `chunks`, `has_openai`, `has_bge_gemma2`, `duration_ms` |
| `funding_extracted` | `fields` — list of metadata keys extracted. Only emitted when `assistant_type = "funding"`. |
| `stored` | `vectors`, `collection` |

### Heartbeats

Every ~15 s while the pipeline is working, the server writes `: keepalive\\n\\n`
(an SSE comment, not an event). Clients can ignore these; they exist only to
prevent reverse proxies from closing idle connections.

---

## Client examples

### Browser (fetch + ReadableStream)

`EventSource` does not support POST, so use `fetch` + a manual SSE parser:

```js
const resp = await fetch("/api/v1/online/ingest/stream", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "your-key",  // only if DP_ONLINE_API_KEYS is configured
  },
  body: JSON.stringify({
    collection_name: "wiener-neudorf",
    source_id: "web_foerderungen_001",
    url: "https://example.gv.at/page",
    content: "...",
    content_type: ["funding"],
    metadata: { assistant_id: "asst_01", municipality_id: "wiener-neudorf" },
  }),
});

const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buf = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += decoder.decode(value, { stream: true });

  // Events are separated by a blank line.
  let idx;
  while ((idx = buf.indexOf("\\n\\n")) !== -1) {
    const raw = buf.slice(0, idx);
    buf = buf.slice(idx + 2);
    if (raw.startsWith(":")) continue;  // heartbeat comment

    const event = {};
    for (const line of raw.split("\\n")) {
      const [k, ...rest] = line.split(":");
      event[k.trim()] = rest.join(":").trim();
    }
    const payload = JSON.parse(event.data);

    if (event.event === "progress") {
      console.log("phase:", payload.phase, payload);
    } else if (event.event === "completed") {
      console.log("done:", payload);
    } else if (event.event === "error") {
      console.error("failed:", payload);
    }
  }
}
```

### Python (httpx, streaming)

```python
import json
import httpx

payload = {
    "collection_name": "wiener-neudorf",
    "source_id": "web_foerderungen_001",
    "url": "https://example.gv.at/page",
    "content": "...",
    "content_type": ["funding"],
    "metadata": {"assistant_id": "asst_01", "municipality_id": "wiener-neudorf"},
}

with httpx.Client(timeout=None) as client:
    with client.stream(
        "POST",
        "http://localhost:8000/api/v1/online/ingest/stream",
        json=payload,
        headers={"X-API-Key": "your-key"},  # only if configured
    ) as resp:
        event_name = None
        for line in resp.iter_lines():
            if line.startswith(":") or line == "":
                event_name = None
                continue
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
                print(event_name, data)
```

### curl

```bash
curl -N -X POST http://localhost:8000/api/v1/online/ingest/stream \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-key" \\
  -d '{"collection_name":"wiener-neudorf","source_id":"doc_1","url":"https://example.gv.at","content":"...","content_type":["funding"],"metadata":{"assistant_id":"asst_01","municipality_id":"wiener-neudorf"}}'
```

The `-N` flag (no buffering) is important — without it curl waits for the
full body before printing, defeating the point of streaming.

---

## Proxy / timeout note

This endpoint keeps one HTTP connection open for the full ingest duration.
If you run a reverse proxy / CDN in front (nginx, Cloudflare, Vercel), raise
its **idle read timeout** above your worst-case ingest time (e.g. 300 s).
Heartbeats every 15 s help but aren't enough if the proxy has a hard
response-time cap.

**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.

**Error codes** (same as `/online/ingest`): `VALIDATION_EMPTY_CONTENT`, `EMBEDDING_MODEL_NOT_LOADED`, `EMBEDDING_FAILED`, `EMBEDDING_OOM`, `QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_UPSERT_FAILED`, `QDRANT_DISK_FULL`.
"""


def _format_sse(event: str, data: dict) -> str:
    """Serialize one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/ingest/stream",
    summary="Stream ingest progress as Server-Sent Events",
    description=SSE_STREAM_DESCRIPTION,
    response_description=(
        "text/event-stream of progress events. Ends with a single `completed` "
        "or `error` event, then the connection closes."
    ),
)
async def ingest_online_stream(body: OnlineIngestRequest, request: Request) -> StreamingResponse:
    request_id = request.state.request_id
    ingest_svc = request.app.state.online_ingest

    async def event_generator() -> AsyncIterator[str]:
        # Fail fast on empty content — no need to stream anything.
        if not body.content.strip():
            yield _format_sse(
                "error",
                {"code": "VALIDATION_EMPTY_CONTENT", "detail": "Content must not be empty"},
            )
            return

        # Kick off the funding extractor in parallel with ingest, same as /ingest.
        funding_task: asyncio.Task | None = None
        if body.assistant_type == "funding":
            extractor = request.app.state.funding_extractor
            funding_task = asyncio.create_task(
                _safe_extract_funding(
                    extractor,
                    body.content,
                    source_url=body.url,
                    country=body.country,
                    source_id=body.source_id,
                )
            )

        chunking = body.chunking
        vcfg = body.vector_config
        metadata_dict = body.metadata.model_dump()
        metadata_dict["source_url"] = body.url
        metadata_dict["assistant_type"] = body.assistant_type
        if body.state_or_province:
            metadata_dict["state_or_province"] = body.state_or_province

        queue: asyncio.Queue = asyncio.Queue()
        ingest_task = asyncio.create_task(
            ingest_svc.ingest(
                source_id=body.source_id,
                file_path=body.url,
                content=body.content,
                acl=None,
                metadata=metadata_dict,
                collection_name=body.collection_name,
                language=body.language,
                chunking_strategy=chunking.strategy if chunking else "contextual",
                max_chunk_size=chunking.max_chunk_size if chunking else None,
                chunk_overlap=chunking.overlap if chunking else None,
                vector_size=vcfg.vector_size if vcfg else 1536,
                search_mode=vcfg.search_mode.value if vcfg else "semantic",
                fallback_dense_dim=ext.bge_gemma2_dense_dim if (vcfg and vcfg.enable_fallback) else None,
                content_type=body.content_type,
                entities=body.entities.model_dump() if body.entities else None,
                deferred_metadata_task=funding_task,
                progress_queue=queue,
            )
        )

        try:
            while True:
                # Race the queue drain against a heartbeat tick and against
                # ingest_task completion (so we don't hang if the pipeline
                # finishes without emitting a terminal queue event).
                get_task = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {get_task, ingest_task},
                    timeout=HEARTBEAT_INTERVAL_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if get_task in done:
                    event = get_task.result()
                    yield _format_sse("progress", event)
                    continue

                # Heartbeat branch — cancel the pending get and send a comment.
                get_task.cancel()
                # If the ingest task finished, break out to emit the terminal event.
                if ingest_task.done():
                    # Drain any remaining queued events first.
                    while not queue.empty():
                        yield _format_sse("progress", queue.get_nowait())
                    break
                yield ": keepalive\n\n"

            # Ingest done — surface success or error.
            try:
                result = ingest_task.result()
            except IngestError as e:
                code_enum = INGEST_ERROR_CODE_MAP.get(e.code)
                code_str = code_enum.value if code_enum is not None else e.code
                log.error(
                    "ingest_stream_failed",
                    source_id=body.source_id,
                    error=str(e),
                    code=e.code,
                    request_id=request_id,
                )
                yield _format_sse("error", {"code": code_str, "detail": str(e)})
                return

            yield _format_sse(
                "completed",
                {
                    "source_id": result.source_id,
                    "chunks_created": result.chunks_created,
                    "vectors_stored": result.vectors_stored,
                    "collection": result.collection,
                    "content_type": result.classification,
                    "embedding_time_ms": result.embedding_time_ms,
                    "total_time_ms": result.total_time_ms,
                },
            )

        except asyncio.CancelledError:
            # Client disconnected — tear down the ingest task.
            log.info("ingest_stream_client_disconnected", source_id=body.source_id, request_id=request_id)
            if not ingest_task.done():
                ingest_task.cancel()
            if funding_task is not None and not funding_task.done():
                funding_task.cancel()
            raise

    # ``X-Accel-Buffering: no`` disables nginx response buffering so events
    # actually reach the client in real time.
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


async def _safe_extract_funding(
    extractor, content: str, *, source_url: str, country: str | None, source_id: str
) -> dict:
    """Run funding extraction; swallow errors so they don't cancel the ingest task."""
    try:
        return await extractor.extract(content, source_url=source_url, country=country)
    except Exception as e:
        log.warning("ingest_stream_funding_extract_failed", source_id=source_id, error=str(e))
        return {}
