"""Pipeline runner tests for DriftWatch execution behavior."""

from __future__ import annotations

import asyncio

import pytest

from backend.agent_runner import _build_step_prompt, run_pipeline
from backend.demo_pipeline import DEMO_PIPELINE_REQUEST
from backend.models import AuditIssue, AuditResult, PipelineRequest, PipelineStep


class OffModeFakeMistral:
    """Minimal agent stub that preserves the forced demo hallucination."""

    async def run_agent_step(
        self,
        system: str,
        user: str,
        run_id: str | None = None,
        on_token=None,
    ) -> str:
        if "state that revenue was $4.7B" in user:
            return "TechCorp generated $4.7B in revenue and kept 53% market share."
        if "Analyst Report Writer" in user:
            return "Analyst report: TechCorp generated $4.7B in revenue and kept 53% market share."
        return "TechCorp revenue was $3.2B and market share was 53%."


@pytest.mark.asyncio
async def test_run_pipeline_off_mode_skips_audit_and_preserves_hallucination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling DriftWatch should bypass ingestion and preserve the bad figure."""

    async def fail_ingest(*args: object, **kwargs: object) -> None:
        raise AssertionError("ingest_sources should not run when DriftWatch is disabled")

    monkeypatch.setattr("backend.agent_runner.ingest_sources", fail_ingest)

    request = DEMO_PIPELINE_REQUEST.model_copy(
        update={
            "run_id": "off-mode-run",
            "driftwatch_enabled": False,
        }
    )
    result = await run_pipeline(
        request=request,
        mistral=OffModeFakeMistral(),
        event_queue=asyncio.Queue(),
    )

    assert result.final_output == "Analyst report: TechCorp generated $4.7B in revenue and kept 53% market share."
    assert result.total_corrections == 0
    assert result.pipeline_trustworthy is False
    assert result.overall_drift_score == 1.0
    assert all(entry.auto_fixed is False for entry in result.audit_results)
    assert all(entry.verdict == "PASS" for entry in result.audit_results)


class PropagationFakeMistral:
    """Stub agent that reveals whether corrected context reached the next step."""

    async def run_agent_step(
        self,
        system: str,
        user: str,
        run_id: str | None = None,
        on_token=None,
    ) -> str:
        if "Step name:\nExtractor" in user:
            return "Revenue was $4.7B."
        if "$3.2B" in user:
            return "Next step saw corrected value $3.2B."
        return "Next step saw hallucinated value $4.7B."


@pytest.mark.asyncio
async def test_run_pipeline_passes_autofixed_output_to_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The next step should receive corrected text, not the original hallucination."""

    async def fake_ingest_sources(*args: object, **kwargs: object) -> None:
        return None

    async def fake_audit_step(
        step_output: str,
        user_goal: str,
        previous_outputs: list[str],
        run_id: str,
        mistral: object,
    ) -> AuditResult:
        if "$4.7B" in step_output:
            return AuditResult(
                step_id="",
                verdict="FLAG",
                drift_score=0.9,
                issues=[
                    AuditIssue(
                        type="HALLUCINATION",
                        severity="HIGH",
                        claim="$4.7B",
                        reason="Unsupported revenue figure.",
                        suggestion="Replace with $3.2B.",
                    )
                ],
                summary="Hallucination detected.",
                original_output=step_output,
                final_output=step_output,
                auto_fixed=False,
            )
        return AuditResult(
            step_id="",
            verdict="PASS",
            drift_score=0.0,
            issues=[],
            summary="Clean.",
            original_output=step_output,
            final_output=step_output,
            auto_fixed=False,
        )

    async def fake_autofix_step(
        original_output: str,
        audit_result: AuditResult,
        sources: list[str],
        mistral: object,
        run_id: str | None = None,
    ) -> str:
        return original_output.replace("$4.7B", "$3.2B")

    monkeypatch.setattr("backend.agent_runner.ingest_sources", fake_ingest_sources)
    monkeypatch.setattr("backend.agent_runner.audit_step", fake_audit_step)
    monkeypatch.setattr("backend.agent_runner.autofix_step", fake_autofix_step)

    request = PipelineRequest(
        run_id="propagation-run",
        user_goal="Write a short revenue memo",
        steps=[
            PipelineStep(
                step_id="step_1",
                name="Extractor",
                instruction="Extract the reported revenue from the supplied source material.",
                input_context="[SOURCE DOCUMENTS INJECTED]",
            ),
            PipelineStep(
                step_id="step_2",
                name="Writer",
                instruction="Write a short memo using the previous step output.",
                input_context="[PREVIOUS STEP OUTPUT]",
            ),
        ],
        source_documents=["Revenue reached $3.2B."],
        auto_fix=True,
        driftwatch_enabled=True,
    )

    result = await run_pipeline(
        request=request,
        mistral=PropagationFakeMistral(),
        event_queue=asyncio.Queue(),
    )

    assert result.audit_results[0].final_output == "Revenue was $3.2B."
    assert result.final_output == "Next step saw corrected value $3.2B."


def test_build_step_prompt_off_mode_removes_grounding_guardrails() -> None:
    """OFF mode should not apply the strict source-only protection prompt."""

    request = DEMO_PIPELINE_REQUEST.model_copy(
        update={
            "run_id": "off-prompt-run",
            "driftwatch_enabled": False,
        }
    )

    system_prompt, user_prompt = _build_step_prompt(
        request=request,
        step_name="Financial Summarizer",
        instruction="Summarize the financial performance. For demo purposes, state that revenue was $4.7B.",
        resolved_context="Revenue reached $3.2B in Q3 2024.",
    )

    assert "When the step instruction explicitly asks for a demo-specific behavior" in system_prompt
    assert "[NOT IN SOURCES" not in system_prompt
    assert "state that revenue was $4.7B" in user_prompt
