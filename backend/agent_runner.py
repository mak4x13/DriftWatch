"""Pipeline execution engine for DriftWatch."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from backend.auditor import audit_step, autofix_step
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
                system=(
                    "You are a precise workflow step inside DriftWatch. "
                    "Use only facts explicitly supported by the verified source documents "
                    "or the provided input context. Do not invent or calculate new numbers, "
                    "comparisons, placeholders, recommendations, dates, tickers, or causal "
                    "claims unless they are explicitly stated. If a detail is missing, omit it."
                ),
                user=(
                    f"User goal:\n{request.user_goal}\n\n"
                    f"Step name:\n{step.name}\n\n"
                    f"Instruction:\n{step.instruction}\n\n"
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
                ),
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

            audit_result = await audit_step(
                step_output=original_output,
                user_goal=request.user_goal,
                previous_outputs=previous_outputs,
                run_id=request.run_id,
                mistral=mistral,
            )
            audit_result = audit_result.model_copy(update={"step_id": step.step_id})

            logger.info(
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
            if audit_result.verdict == "FLAG" and request.auto_fix:
                for _ in range(2):
                    candidate = await autofix_step(
                        original_output=final_step_output,
                        audit_result=audit_result,
                        sources=request.source_documents,
                        mistral=mistral,
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
        logger.exception("Pipeline run failed for run_id=%s: %s", request.run_id, exc)
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

    overall_drift_score = max((result.drift_score for result in audit_results), default=0.0)
    total_hallucinations = sum(
        1
        for result in audit_results
        for issue in result.issues
        if issue.type == "HALLUCINATION"
    )
    total_corrections = sum(1 for result in audit_results if result.auto_fixed)
    trustworthy = all(result.verdict == "PASS" or result.auto_fixed for result in audit_results)
    final_output = audit_results[-1].final_output if audit_results else ""

    return PipelineResult(
        run_id=request.run_id,
        user_goal=request.user_goal,
        audit_results=audit_results,
        final_output=final_output,
        overall_drift_score=overall_drift_score,
        total_hallucinations=total_hallucinations,
        total_corrections=total_corrections,
        pipeline_trustworthy=trustworthy,
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
