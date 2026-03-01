"""Tests for DriftWatch auditing heuristics and autofix behavior."""

from __future__ import annotations

import pytest

from backend.agent_runner import _build_pipeline_result
from backend.auditor import _adjust_drift_score, _extract_claim_candidates, audit_step, autofix_step
from backend.models import AuditIssue, AuditResult, PipelineRequest, PipelineStep


class FakeMistral:
    """Minimal async stub for auditor and autofix tests."""

    async def run_auditor(
        self, audit_prompt: str, run_id: str | None = None
    ) -> AuditResult:
        return AuditResult(
            step_id="",
            verdict="PASS",
            drift_score=0.0,
            issues=[],
            summary="Fallback pass.",
            original_output="",
            final_output="",
            auto_fixed=False,
        )

    async def run_autofix(
        self,
        original: str,
        issues: list[AuditIssue],
        sources: str,
        run_id: str | None = None,
    ) -> str:
        for issue in issues:
            if issue.claim in original and "$3.2B" in issue.suggestion:
                return original.replace(issue.claim, "$3.2B")
        return original


def test_extract_claim_candidates_finds_numeric_sentences() -> None:
    """Claim extraction should keep metric-heavy sentences for grounding checks."""

    text = "Revenue reached $4.7B. The CEO sounded optimistic. Market share hit 53%."
    claims = _extract_claim_candidates(text)
    assert "Revenue reached $4.7B." in claims
    assert "Market share hit 53%." in claims


@pytest.mark.asyncio
async def test_audit_step_flags_hallucinated_revenue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit should flag an unsupported revenue figure against the provided sources."""

    async def fake_query_sources(claim: str, run_id: str, n_results: int = 3) -> list[str]:
        return [
            "TechCorp Q3 2024 Earnings: Revenue reached $3.2B, up 12% YoY.",
            "Competitor Report: TechCorp maintains 53% market share in enterprise segment.",
        ]

    monkeypatch.setattr("backend.auditor.query_sources", fake_query_sources)

    result = await audit_step(
        step_output="TechCorp delivered $4.7B in revenue and 53% market share.",
        user_goal="Analyze TechCorp Q3 performance and write analyst report",
        previous_outputs=["Key fact: TechCorp revenue reached $3.2B in Q3 2024."],
        run_id="demo-run",
        mistral=FakeMistral(),
    )

    assert result.verdict == "FLAG"
    assert any(issue.type == "HALLUCINATION" for issue in result.issues)
    assert "$4.7B" in result.summary


@pytest.mark.asyncio
async def test_autofix_step_replaces_unsupported_claim() -> None:
    """Autofix should replace the unsupported figure with the sourced figure."""

    audit_result = AuditResult(
        step_id="step_2",
        verdict="FLAG",
        drift_score=0.75,
        issues=[
            AuditIssue(
                type="HALLUCINATION",
                severity="HIGH",
                claim="$4.7B",
                reason="Claim '$4.7B' does not appear in the retrieved sources.",
                suggestion="Replace the unsupported figure with the sourced figure $3.2B.",
            )
        ],
        summary="Hallucination detected: $4.7B not found in sources (actual: $3.2B).",
        original_output="TechCorp delivered $4.7B in revenue.",
        final_output="TechCorp delivered $4.7B in revenue.",
        auto_fixed=False,
    )

    corrected = await autofix_step(
        original_output="TechCorp delivered $4.7B in revenue.",
        audit_result=audit_result,
        sources=["TechCorp Q3 2024 Earnings: Revenue reached $3.2B, up 12% YoY."],
        mistral=FakeMistral(),
    )

    assert "$3.2B" in corrected
    assert "$4.7B" not in corrected


def test_build_pipeline_result_uses_post_fix_drift_for_trust() -> None:
    """Auto-fixed steps should contribute a reduced effective drift score."""

    request = PipelineRequest(
        user_goal="Analyze a financial report",
        steps=[
            PipelineStep(
                step_id="step_1",
                name="Summarizer",
                instruction="Summarize the financial report using only grounded figures.",
                input_context="[SOURCE DOCUMENTS INJECTED]",
            )
        ],
        source_documents=["Revenue reached $3.2B."],
        auto_fix=True,
        driftwatch_enabled=True,
    )
    result = _build_pipeline_result(
        request,
        [
            AuditResult(
                step_id="step_1",
                step_name="Summarizer",
                verdict="FLAG",
                drift_score=0.75,
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
                original_output="Revenue was $4.7B.",
                final_output="Revenue was $3.2B.",
                auto_fixed=True,
            )
        ],
    )

    assert result.overall_drift_score == 0.1
    assert result.pipeline_trustworthy is True
    assert result.overall_verdict == "TRUSTWORTHY"
    assert result.total_corrections == 1


def test_adjusted_drift_score_scales_with_issue_volume() -> None:
    """Many medium/high issues should sharply increase the post-processed drift score."""

    issues = [
        AuditIssue(
            type="HALLUCINATION",
            severity="HIGH",
            claim=f"${index}.0M",
            reason="Unsupported figure.",
            suggestion="Replace with a sourced value.",
        )
        for index in range(4)
    ] + [
        AuditIssue(
            type="CONTRADICTION",
            severity="MEDIUM",
            claim="Revenue claim mismatch.",
            reason="Conflicts with prior output.",
            suggestion="Use the sourced value.",
        )
    ]

    adjusted = _adjust_drift_score(0.15, issues)

    assert adjusted == 0.5


@pytest.mark.asyncio
async def test_audit_step_single_high_issue_does_not_overstate_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single grounded hallucination flag should not spike drift into catastrophic range."""

    async def fake_query_sources(claim: str, run_id: str, n_results: int = 3) -> list[str]:
        return ["TechCorp Q3 2024 Earnings: Revenue reached $3.2B, up 12% YoY."]

    monkeypatch.setattr("backend.auditor.query_sources", fake_query_sources)

    result = await audit_step(
        step_output="TechCorp delivered $4.7B in revenue.",
        user_goal="Analyze TechCorp Q3 performance and write analyst report",
        previous_outputs=[],
        run_id="single-high-run",
        mistral=FakeMistral(),
    )

    assert result.verdict == "FLAG"
    assert result.drift_score == 0.63


@pytest.mark.asyncio
async def test_autofix_step_repairs_malformed_percentage_rendering() -> None:
    """Autofix should repair concatenated business percentages such as 1211%."""

    class BrokenPercentMistral:
        async def run_autofix(
            self,
            original: str,
            issues: list[AuditIssue],
            sources: str,
            run_id: str | None = None,
        ) -> str:
            return "Net Revenue Retention (NRR): 1211%"

    audit_result = AuditResult(
        step_id="step_2",
        verdict="FLAG",
        drift_score=0.75,
        issues=[
            AuditIssue(
                type="HALLUCINATION",
                severity="HIGH",
                claim="11%",
                reason="Unsupported retention figure.",
                suggestion="Replace with 124%.",
            )
        ],
        summary="Hallucination detected.",
        original_output="Net Revenue Retention (NRR): 11%",
        final_output="Net Revenue Retention (NRR): 11%",
        auto_fixed=False,
    )

    corrected = await autofix_step(
        original_output="Net Revenue Retention (NRR): 11%",
        audit_result=audit_result,
        sources=["Net Revenue Retention (NRR): 124%"],
        mistral=BrokenPercentMistral(),
    )

    assert corrected == "Net Revenue Retention (NRR): 124%"


def test_build_pipeline_result_marks_partial_for_mid_drift_unresolved_issue() -> None:
    """Mid-drift runs with unresolved issues should land in the partial state."""

    request = PipelineRequest(
        user_goal="Analyze a financial report",
        steps=[
            PipelineStep(
                step_id="step_1",
                name="Summarizer",
                instruction="Summarize the financial report using only grounded figures.",
                input_context="[SOURCE DOCUMENTS INJECTED]",
            )
        ],
        source_documents=["Revenue reached $3.2B."],
        auto_fix=True,
        driftwatch_enabled=True,
    )
    result = _build_pipeline_result(
        request,
        [
            AuditResult(
                step_id="step_1",
                step_name="Summarizer",
                verdict="FLAG",
                drift_score=0.35,
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
                original_output="Revenue was $4.7B.",
                final_output="Revenue was $4.7B.",
                auto_fixed=False,
            )
        ],
    )

    assert result.pipeline_trustworthy is False
    assert result.overall_verdict == "PARTIALLY_VERIFIED"


def test_build_pipeline_result_requires_review_when_corrections_lag_hallucinations() -> None:
    """Trust must fail when not every flagged hallucination is corrected."""

    request = PipelineRequest(
        user_goal="Analyze a startup memo",
        steps=[
            PipelineStep(
                step_id="step_1",
                name="Memo Writer",
                instruction="Write the memo using only grounded figures from the source materials.",
                input_context="[SOURCE DOCUMENTS INJECTED]",
            )
        ],
        source_documents=["Growth was 180%, burn was $185,000, concentration was 18%."],
        auto_fix=True,
        driftwatch_enabled=True,
    )
    result = _build_pipeline_result(
        request,
        [
            AuditResult(
                step_id="step_1",
                step_name="Memo Writer",
                verdict="FLAG",
                drift_score=1.0,
                issues=[
                    AuditIssue(
                        type="HALLUCINATION",
                        severity="HIGH",
                        claim="110%",
                        reason="Unsupported growth figure.",
                        suggestion="Replace with 180%.",
                    ),
                    AuditIssue(
                        type="HALLUCINATION",
                        severity="HIGH",
                        claim="$115,000",
                        reason="Unsupported burn figure.",
                        suggestion="Replace with $185,000.",
                    ),
                    AuditIssue(
                        type="HALLUCINATION",
                        severity="HIGH",
                        claim="11%",
                        reason="Unsupported concentration figure.",
                        suggestion="Replace with 18%.",
                    ),
                ],
                summary="Hallucinations detected.",
                original_output="Growth 110%, burn $115,000, concentration 11%.",
                final_output="Growth 180%, burn $115,000, concentration 11%.",
                auto_fixed=False,
                detected_hallucinations=3,
                corrected_hallucinations=1,
            )
        ],
    )

    assert result.total_hallucinations == 3
    assert result.total_corrections == 1
    assert result.pipeline_trustworthy is False
    assert result.overall_verdict == "REVIEW_REQUIRED"


def test_build_pipeline_result_marks_partial_when_not_in_sources_remains() -> None:
    """Presence of a NOT IN SOURCES marker should prevent full trust."""

    request = PipelineRequest(
        user_goal="Write a startup memo",
        steps=[
            PipelineStep(
                step_id="step_1",
                name="Memo Writer",
                instruction="Write a grounded memo using only the provided source materials.",
                input_context="[SOURCE DOCUMENTS INJECTED]",
            )
        ],
        source_documents=["ARR is $1.8M."],
        auto_fix=True,
        driftwatch_enabled=True,
    )
    result = _build_pipeline_result(
        request,
        [
            AuditResult(
                step_id="step_1",
                step_name="Memo Writer",
                verdict="PASS",
                drift_score=0.2,
                issues=[],
                summary="Grounded output.",
                original_output="ARR is $1.8M.",
                final_output="[NOT IN SOURCES: this information was not provided in the source documents]",
                auto_fixed=False,
            )
        ],
    )

    assert result.pipeline_trustworthy is False
    assert result.overall_verdict == "PARTIALLY_VERIFIED"
