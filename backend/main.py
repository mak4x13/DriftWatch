"""FastAPI entry point for the DriftWatch backend."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from backend.agent_runner import run_pipeline
from backend.mistral_client import MistralClient
from backend.models import AuditEvent, PipelineRequest, PipelineResult, SourceDocument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb.api.segment").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
RUN_QUEUES: dict[str, asyncio.Queue] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services for the DriftWatch application."""

    app.state.mistral = MistralClient()
    try:
        from backend.vector_store import init_chroma

        init_chroma()
    except Exception as exc:
        logger.warning("Chroma initialization failed at startup: %s", exc)
    yield
    RUN_QUEUES.clear()


app = FastAPI(title="DriftWatch", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.post("/api/pipeline/run", response_model=PipelineResult)
async def run_pipeline_route(request: PipelineRequest) -> PipelineResult:
    """Run a custom pipeline request and return the completed result."""

    queue = _get_or_create_queue(request.run_id)
    try:
        return await run_pipeline(
            request=request,
            mistral=_get_mistral_client(),
            event_queue=queue,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/pipeline/stream/{run_id}")
async def stream_pipeline_events(run_id: str, request: Request) -> EventSourceResponse:
    """Stream audit events for a pipeline run via server-sent events."""

    queue = _get_or_create_queue(run_id)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event: AuditEvent = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    continue
                yield {"event": "message", "data": event.model_dump_json()}
                if event.event_type in {"PIPELINE_DONE", "ERROR"}:
                    break
        finally:
            RUN_QUEUES.pop(run_id, None)

    return EventSourceResponse(event_generator())


@app.post("/api/pipeline/demo", response_model=PipelineResult)
async def run_demo_pipeline(
    run_id: str | None = Query(default=None),
) -> PipelineResult:
    """Run the prebuilt demo pipeline used in the hackathon presentation."""

    from backend.demo_pipeline import DEMO_PIPELINE_REQUEST

    request = (
        DEMO_PIPELINE_REQUEST.model_copy(update={"run_id": run_id})
        if run_id
        else DEMO_PIPELINE_REQUEST
    )
    queue = _get_or_create_queue(request.run_id)
    try:
        return await run_pipeline(
            request=request,
            mistral=_get_mistral_client(),
            event_queue=queue,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/sources/ingest")
async def ingest_source_documents(documents: list[SourceDocument]) -> dict[str, object]:
    """Ingest a batch of source documents into the Chroma collection."""

    if not documents:
        raise HTTPException(status_code=400, detail="At least one source document is required.")

    from backend.vector_store import ingest_sources

    run_id = documents[0].run_id
    await ingest_sources([document.text for document in documents], run_id)
    return {"status": "ok", "run_id": run_id, "ingested_count": len(documents)}


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """Report whether the app, Mistral client, and Chroma store are available."""

    try:
        from backend.vector_store import init_chroma

        init_chroma()
        chroma_status = "connected"
    except Exception:
        chroma_status = "disconnected"

    mistral_status = "connected" if _get_mistral_client().client is not None else "not_configured"
    return {
        "status": "ok",
        "mistral": mistral_status,
        "chromadb": chroma_status,
    }


@app.get("/")
async def serve_frontend() -> FileResponse:
    """Serve the single-page DriftWatch frontend."""

    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css")
async def serve_stylesheet() -> FileResponse:
    """Serve the frontend stylesheet for direct index.html loading."""

    return FileResponse(FRONTEND_DIR / "styles.css")


@app.get("/app.js")
async def serve_app_script() -> FileResponse:
    """Serve the frontend script for direct index.html loading."""

    return FileResponse(FRONTEND_DIR / "app.js")


@app.get("/favicon.ico")
async def serve_favicon() -> Response:
    """Return an empty favicon response to keep browser requests quiet."""

    return Response(status_code=204)


def _get_or_create_queue(run_id: str) -> asyncio.Queue:
    """Return the existing SSE queue for a run or create a new one."""

    queue = RUN_QUEUES.get(run_id)
    if queue is None:
        queue = asyncio.Queue()
        RUN_QUEUES[run_id] = queue
    return queue


def _get_mistral_client() -> MistralClient:
    """Return the shared Mistral client, creating it if lifespan has not run yet."""

    mistral = getattr(app.state, "mistral", None)
    if mistral is None:
        mistral = MistralClient()
        app.state.mistral = mistral
    return mistral
