from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Status of each external dependency checked during readiness."""

    qdrant: bool = Field(False, description="Qdrant vector database reachable")
    bge_m3: bool = Field(False, description="BGE-M3 embedding model loaded and responding")
    openai_embedder: bool = Field(False, description="OpenAI embeddings API reachable (primary online embedder)")
    bge_gemma2: bool = Field(False, description="BGE-Gemma2 via LiteLLM reachable (fallback online embedder)")
    parser: bool = Field(False, description="Document parser available (LlamaParse or Unstructured)")
    crawl4ai: bool = Field(False, description="Crawl4AI web scraping service available")
    ldap: bool = Field(False, description="LDAP/Active Directory server reachable")
    redis: bool = Field(False, description="Redis cache reachable")


class ModelHealthItem(BaseModel):
    """Health of a single model-bearing component."""

    component: str = Field(..., description="Internal component name")
    task: str = Field(..., description="What this component is used for")
    provider: str = Field(..., description="Provider or serving stack")
    model: str = Field(..., description="Configured model name or backend label")
    healthy: bool = Field(..., description="True if the component is configured and responding")
    configured: bool = Field(..., description="True if the component is configured or enabled")
    required: bool = Field(..., description="True if this configured component is expected to be healthy")
    detail: str | None = Field(None, description="Short explanation for disabled or unhealthy state")


class HealthResponse(BaseModel):
    """Liveness check response. Returns immediately if the process is running."""

    status: str = Field("ok", description="Always 'ok' if the service is alive")
    uptime_seconds: float | None = Field(None, description="Seconds since service started")

    model_config = {
        "json_schema_extra": {
            "examples": [{"status": "ok", "uptime_seconds": 3421.5}]
        }
    }


class ReadyResponse(BaseModel):
    """Readiness check response.

    Without HMAC auth headers: returns minimal `{ready: true}`.
    With HMAC auth headers (X-Signature): returns full dependency status.
    """

    ready: bool = Field(..., description="True if all core services are operational")
    services: ServiceStatus | None = Field(None, description="Per-service health (only with HMAC auth)")
    mode: str | None = Field(None, description="Deployment mode: on-premise or cloud")
    tenant_id: str | None = Field(None, description="Municipality/tenant identifier")
    worker_id: str | None = Field(None, description="Worker instance identifier")
    version: str | None = Field(None, description="Data Plane version")
    uptime_seconds: float | None = Field(None, description="Seconds since service started")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ready": True,
                    "services": {
                        "qdrant": True,
                        "bge_m3": True,
                        "openai_embedder": True,
                        "bge_gemma2": True,
                        "parser": True,
                        "crawl4ai": True,
                        "ldap": True,
                        "redis": True,
                    },
                    "mode": "on-premise",
                    "tenant_id": "wiener-neudorf",
                    "worker_id": "wn-worker-01",
                    "version": "1.0.0",
                    "uptime_seconds": 3421.5,
                }
            ]
        }
    }


class ModelHealthResponse(BaseModel):
    """Detailed model health response."""

    healthy: bool = Field(..., description="True if all configured model components are healthy")
    models: list[ModelHealthItem] = Field(default_factory=list, description="Per-model component health entries")
    mode: str | None = Field(None, description="Deployment mode: on-premise or cloud")
    tenant_id: str | None = Field(None, description="Municipality/tenant identifier")
    worker_id: str | None = Field(None, description="Worker instance identifier")
    version: str | None = Field(None, description="Data Plane version")
    uptime_seconds: float | None = Field(None, description="Seconds since service started")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "healthy": True,
                    "models": [
                        {
                            "component": "local_embedding",
                            "task": "local document embeddings",
                            "provider": "bge-m3 service",
                            "model": "bge-m3",
                            "healthy": True,
                            "configured": True,
                            "required": True,
                            "detail": None,
                        },
                        {
                            "component": "online_embedding_primary",
                            "task": "online embedding and query embedding",
                            "provider": "openai",
                            "model": "text-embedding-3-small",
                            "healthy": True,
                            "configured": True,
                            "required": True,
                            "detail": None,
                        },
                    ],
                    "mode": "on-premise",
                    "tenant_id": "wiener-neudorf",
                    "worker_id": "wn-worker-01",
                    "version": "1.0.0",
                    "uptime_seconds": 3421.5,
                }
            ]
        }
    }
