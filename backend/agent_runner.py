"""Pipeline execution engine for DriftWatch."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from backend.auditor import audit_step, autofix_step
from backend.logging_utils import get_run_logger
from backend.models import AuditEvent, AuditResult, PipelineRequest, PipelineResult
from backend.mistral_client import MistralClient
from backend.vector_store import ingest_sources

logger = logging.getLogger(__name__)


async def run_pipeline(
    request: PipelineRequest,
    mistral: MistralClient,
    event_queue: asyncio.Queue,
) -> PipelineResult:
    """Execute a pipeline end-to-end with audit checks after each step."""

    run_logger = get_run_logger(logger, request.run_id)
    audit_results: list[AuditResult] = []
    previous_outputs: list[str] = []
    final_output = ""

    try:
        await ingest_sources(request.source_documents, request.run_id)

        for step in request.steps:
            resolved_context = _resolve_input_context(
                template=step.input_context,
                source_documents=request.source_documents,
                previous_output=final_output,
            )
            system_prompt, user_prompt = _build_step_prompt(
                request=request,
                step_name=step.name,
                instruction=step.instruction,
                resolved_context=resolved_context,
            )
            await _emit_event(
                event_queue=event_queue,
                event_type="STEP_START",
                step_id=step.step_id,
                data={
                    "step_name": step.name,
                    "instruction": step.instruction,
                },
            )

            original_output = await mistral.run_agent_step(
                system=system_prompt,
                user=user_prompt,
                run_id=request.run_id,
            )

            await _emit_event(
                event_queue=event_queue,
                event_type="STEP_COMPLETE",
                step_id=step.step_id,
                data={
                    "step_name": step.name,
                    "output_preview": original_output[:200],
                    "output": original_output,
                },
            )

            if request.driftwatch_enabled:
                audit_result = await audit_step(
                    step_output=original_output,
                    user_goal=request.user_goal,
                    previous_outputs=previous_outputs,
                    run_id=request.run_id,
                    mistral=mistral,
                )
                audit_result = audit_result.model_copy(
                    update={
                        "step_id": step.step_id,
                        "step_name": step.name,
                    }
                )
            else:
                audit_result = AuditResult(
                    step_id=step.step_id,
                    step_name=step.name,
                    verdict="PASS",
                    drift_score=1.0,
                    issues=[],
                    summary="DriftWatch protection disabled. Output forwarded without audit.",
                    original_output=original_output,
                    final_output=original_output,
                    auto_fixed=False,
                )

            run_logger.info(
                "Audit decision step_id=%s verdict=%s drift_score=%.2f",
                step.step_id,
                audit_result.verdict,
                audit_result.drift_score,
            )
            await _emit_event(
                event_queue=event_queue,
                event_type="AUDIT_RESULT",
                step_id=step.step_id,
                data=audit_result.model_dump(),
            )

            final_step_output = original_output
            auto_fixed = False
            if (
                request.driftwatch_enabled
                and audit_result.verdict == "FLAG"
                and request.auto_fix
            ):
                for _ in range(2):
                    candidate = await autofix_step(
                        original_output=final_step_output,
                        audit_result=audit_result,
                        sources=request.source_documents,
                        mistral=mistral,
                        run_id=request.run_id,
                    )
                    if candidate.strip() and candidate.strip() != final_step_output.strip():
                        final_step_output = candidate.strip()
                        auto_fixed = True
                        break

                if auto_fixed:
                    audit_result = audit_result.model_copy(
                        update={
                            "final_output": final_step_output,
                            "auto_fixed": True,
                        }
                    )
                    await _emit_event(
                        event_queue=event_queue,
                        event_type="AUTOFIX",
                        step_id=step.step_id,
                        data={
                            "step_name": step.name,
                            "summary": _build_autofix_summary(audit_result),
                            "original_output": original_output,
                            "final_output": final_step_output,
                        },
                    )
                else:
                    audit_result = audit_result.model_copy(
                        update={"final_output": final_step_output}
                    )
            else:
                audit_result = audit_result.model_copy(
                    update={"final_output": final_step_output}
                )

            audit_results.append(audit_result)
            final_output = final_step_output
            previous_outputs.append(final_step_output)

        result = _build_pipeline_result(request=request, audit_results=audit_results)
        await _emit_event(
            event_queue=event_queue,
            event_type="PIPELINE_DONE",
            step_id="pipeline",
            data=result.model_dump(),
        )
        return result
    except Exception as exc:
        run_logger.exception("Pipeline run failed: %s", exc)
        await _emit_event(
            event_queue=event_queue,
            event_type="ERROR",
            step_id="pipeline",
            data={"message": str(exc), "run_id": request.run_id},
        )
        raise


def _resolve_input_context(
    template: str, source_documents: list[str], previous_output: str
) -> str:
    """Resolve step placeholder tokens into concrete context text."""

    resolved = template
    resolved = resolved.replace("[SOURCE DOCUMENTS INJECTED]", "\n\n".join(source_documents))
    resolved = resolved.replace("[PREVIOUS STEP OUTPUT]", previous_output or "No previous output.")
    return resolved


def _build_step_prompt(
    request: PipelineRequest,
    step_name: str,
    instruction: str,
    resolved_context: str,
) -> tuple[str, str]:
    """Build the agent prompt, with stricter grounding only when protection is enabled."""

    if request.driftwatch_enabled:
        system_prompt = (
            "You are a precise workflow step inside DriftWatch. "
            "Use only facts explicitly supported by the verified source documents "
            "or the provided input context. Do not invent or calculate new numbers, "
            "comparisons, placeholders, recommendations, dates, tickers, or causal "
            "claims unless they are explicitly stated. If a detail is missing, omit it."
        )
        user_prompt = (
            f"User goal:\n{request.user_goal}\n\n"
            f"Step name:\n{step_name}\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Input context:\n{resolved_context}\n\n"
            f"Verified source documents:\n{chr(10).join(request.source_documents)}\n\n"
            "Output rules:\n"
            "- Use only grounded facts from the input context or verified sources.\n"
            "- Do not invent arithmetic differences or comparative deltas.\n"
            "- Do not add placeholders such as [Insert Date], [Ticker], [X], or [Your Name].\n"
            "- If a requested field is not present in the sources, leave it out.\n"
            "- Preserve the stated figures and timelines exactly when possible.\n"
            "- If the instruction explicitly says a demo hallucination is required, "
            "include only that one intentional incorrect claim and keep everything else grounded.\n"
        )
        return system_prompt, user_prompt

    system_prompt = (
        "You are a workflow step in a multi-stage AI pipeline. "
        "Follow the instruction using the supplied input context."
    )
    user_prompt = (
        f"User goal:\n{request.user_goal}\n\n"
        f"Step name:\n{step_name}\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Input context:\n{resolved_context}\n"
    )
    return system_prompt, user_prompt


def _build_autofix_summary(audit_result: AuditResult) -> str:
    """Build a human-readable autofix summary for the event stream."""

    for issue in audit_result.issues:
        replacement = _extract_replacement(issue.suggestion)
        if issue.type == "HALLUCINATION" and replacement:
            return f"Fixed: replaced {issue.claim} with {replacement} per source document."
    return "Applied an automatic correction based on the flagged audit issues."


def _extract_replacement(suggestion: str) -> str | None:
    """Extract the first replacement token from an issue suggestion."""

    import re

    match = re.search(r"\$?\d+(?:\.\d+)?(?:B|M|%|x)?", suggestion)
    return match.group(0) if match else None


def _build_pipeline_result(
    request: PipelineRequest, audit_results: list[AuditResult]
) -> PipelineResult:
    """Aggregate final pipeline metrics from per-step audit results."""

    effective_scores = [
        0.1 if result.auto_fixed else result.drift_score
        for result in audit_results
    ]
    overall_drift_score = (
        1.0
        if not request.driftwatch_enabled
        else max(effective_scores, default=0.0)
    )
    total_hallucinations = sum(
        1
        for result in audit_results
        for issue in result.issues
        if issue.type == "HALLUCINATION"
    )
    total_corrections = sum(1 for result in audit_results if result.auto_fixed)
    unresolved_high = any(
        not result.auto_fixed
        and any(issue.severity == "HIGH" for issue in result.issues)
        for result in audit_results
    )
    trustworthy = (
        request.driftwatch_enabled
        and overall_drift_score < 0.4
        and not unresolved_high
    )
    final_output = audit_results[-1].final_output if audit_results else ""
    overall_verdict = "TRUSTWORTHY" if trustworthy else "REVIEW_REQUIRED"

    return PipelineResult(
        run_id=request.run_id,
        user_goal=request.user_goal,
        audit_results=audit_results,
        final_output=final_output,
        overall_drift_score=overall_drift_score,
        total_hallucinations=total_hallucinations,
        total_corrections=total_corrections,
        pipeline_trustworthy=trustworthy,
        overall_verdict=overall_verdict,
        driftwatch_enabled=request.driftwatch_enabled,
    )


async def _emit_event(
    event_queue: asyncio.Queue,
    event_type: str,
    step_id: str,
    data: dict,
) -> None:
    """Push a timestamped audit event onto the SSE queue."""

    event = AuditEvent(
        event_type=event_type,
        step_id=step_id,
        data=data,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    await event_queue.put(event)
