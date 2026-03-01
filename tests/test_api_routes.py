"""Route-level tests for DriftWatch's FastAPI surface."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from backend.main import _get_or_create_queue, app
from backend.models import AuditEvent, AuditResult, GeneratedPipelineStep, PipelineResult


class FakeRouteMistral:
    """Minimal route stub for health and step-generation endpoints."""

    client = object()

    async def generate_pipeline_steps(
        self,
        user_goal: str,
        source_documents: list[str],
        run_id: str | None = None,
    ) -> list[GeneratedPipelineStep]:
        return [
            GeneratedPipelineStep(
                step_name="Source Extractor",
                instruction="Extract the most important grounded facts from the provided sources.",
            ),
            GeneratedPipelineStep(
                step_name="Final Writer",
                instruction="Write the final answer using only the grounded facts from the sources.",
            ),
        ]


def _build_result(run_id: str, user_goal: str) -> PipelineResult:
    """Create a stable pipeline result payload for route tests."""

    return PipelineResult(
        run_id=run_id,
        user_goal=user_goal,
        audit_results=[
            AuditResult(
                step_id="step_1",
                step_name="Source Extractor",
                verdict="PASS",
                drift_score=0.1,
                issues=[],
                summary="Clean output.",
                original_output="Grounded facts.",
                final_output="Grounded facts.",
                auto_fixed=False,
                detected_hallucinations=0,
                corrected_hallucinations=0,
            )
        ],
        final_output="Grounded facts.",
        overall_drift_score=0.1,
        total_hallucinations=0,
        total_corrections=0,
        pipeline_trustworthy=True,
        overall_verdict="TRUSTWORTHY",
        driftwatch_enabled=True,
    )


def test_static_and_health_routes(monkeypatch) -> None:
    """Root, static assets, favicon, and health should all respond successfully."""

    monkeypatch.setattr("backend.main._get_mistral_client", lambda: FakeRouteMistral())
    monkeypatch.setattr("backend.vector_store.init_chroma", lambda: object())

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/styles.css").status_code == 200
        assert client.get("/app.js").status_code == 200
        assert client.get("/favicon.ico").status_code == 204

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "mistral": "connected",
            "chromadb": "connected",
        }


def test_generate_steps_route(monkeypatch) -> None:
    """The step generation route should return AI-generated pipeline steps."""

    monkeypatch.setattr("backend.main._get_mistral_client", lambda: FakeRouteMistral())

    with TestClient(app) as client:
        response = client.post(
            "/api/pipeline/generate-steps",
            json={
                "run_id": "generate-run",
                "user_goal": "Analyze a report",
                "source_documents": ["Revenue reached $3.2B."],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "generate-run"
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["step_name"] == "Source Extractor"


def test_run_and_demo_routes(monkeypatch) -> None:
    """Custom run and demo endpoints should return the pipeline result contract."""

    async def fake_run_pipeline(request, mistral, event_queue):
        return _build_result(request.run_id, request.user_goal)

    monkeypatch.setattr("backend.main.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("backend.main._get_mistral_client", lambda: FakeRouteMistral())

    with TestClient(app) as client:
        run_response = client.post(
            "/api/pipeline/run",
            json={
                "run_id": "route-run",
                "user_goal": "Analyze a report",
                "steps": [
                    {
                        "step_id": "step_1",
                        "name": "Writer",
                        "instruction": "Write a grounded answer using only the supplied source material.",
                        "input_context": "[SOURCE DOCUMENTS INJECTED]",
                    }
                ],
                "source_documents": ["Revenue reached $3.2B."],
                "auto_fix": True,
                "driftwatch_enabled": True,
            },
        )
        demo_response = client.post("/api/pipeline/demo?run_id=demo-route")

    assert run_response.status_code == 200
    assert run_response.json()["run_id"] == "route-run"
    assert demo_response.status_code == 200
    assert demo_response.json()["run_id"] == "demo-route"


def test_ingest_reset_and_stream_routes(monkeypatch) -> None:
    """Ingest, reset, and SSE stream routes should all work with deterministic data."""

    captured: dict[str, object] = {}

    async def fake_ingest_sources(texts: list[str], run_id: str) -> None:
        captured["texts"] = texts
        captured["run_id"] = run_id

    monkeypatch.setattr("backend.vector_store.ingest_sources", fake_ingest_sources)
    monkeypatch.setattr(
        "backend.vector_store.reset_source_collections",
        lambda: ["sources_1024"],
    )

    run_id = "stream-route"
    queue = _get_or_create_queue(run_id)
    asyncio.run(
        queue.put(
            AuditEvent(
                event_type="PIPELINE_DONE",
                step_id="pipeline",
                data=_build_result(run_id, "Analyze a report").model_dump(),
                timestamp="2026-03-01T00:00:00+00:00",
            )
        )
    )

    with TestClient(app) as client:
        ingest_response = client.post(
            "/api/sources/ingest",
            json=[{"text": "Revenue reached $3.2B.", "run_id": "ingest-run"}],
        )
        with client.stream("GET", f"/api/pipeline/stream/{run_id}") as response:
            stream_body = "".join(response.iter_text())
        reset_response = client.post("/api/demo/reset")

    assert ingest_response.status_code == 200
    assert ingest_response.json()["ingested_count"] == 1
    assert captured == {
        "texts": ["Revenue reached $3.2B."],
        "run_id": "ingest-run",
    }

    assert reset_response.status_code == 200
    assert reset_response.json()["deleted_collections"] == ["sources_1024"]

    assert '"event_type":"PIPELINE_DONE"' in stream_body
    assert '"run_id":"stream-route"' in stream_body
