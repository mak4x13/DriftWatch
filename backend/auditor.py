"""Semantic audit and autofix logic for DriftWatch pipeline steps."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from backend.mistral_client import MistralClient
from backend.models import AuditIssue, AuditResult
from backend.vector_store import query_sources

logger = logging.getLogger(__name__)

AUDITOR_SYSTEM_PROMPT = """
You are DriftWatch, a semantic audit engine for AI pipelines.
Your job is to evaluate AI-generated outputs for three failure modes:
1. INTENT DRIFT: Does this output still serve the original user goal?
2. HALLUCINATION: Is every specific claim (numbers, names, dates,
   statistics) traceable to the provided source documents?
3. CONTRADICTION: Does this output contradict any previous step outputs?
Respond ONLY in this exact JSON format:
{  "verdict": "PASS" or "FLAG",
   "drift_score": 0.0-1.0,
   "issues": [{"type": "...", "severity": "...",
               "claim": "...", "reason": "...", "suggestion": "..."}],
   "summary": "one sentence plain English summary" }
Flag if drift_score > 0.3 or any HIGH severity issue exists.
"""

NUMERIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\$?\d+(?:\.\d+)?(?:\s*(?:B|M|%|x|billion|million|percent|times))?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
KEYWORD_PATTERN = re.compile(
    r"\b(revenue|earnings|market share|inflation|price target|p/e|r&d|growth)\b",
    re.IGNORECASE,
)


async def audit_step(
    step_output: str,
    user_goal: str,
    previous_outputs: list[str],
    run_id: str,
    mistral: MistralClient,
) -> AuditResult:
    """Audit one pipeline step against intent, sources, and prior outputs."""

    source_passages = await _gather_source_context(step_output, run_id)
    audit_prompt = _build_audit_prompt(
        step_output=step_output,
        user_goal=user_goal,
        previous_outputs=previous_outputs,
        source_passages=source_passages,
    )
    llm_result = await mistral.run_auditor(audit_prompt)
    heuristic_result = _run_rule_based_audit(
        step_output=step_output,
        user_goal=user_goal,
        previous_outputs=previous_outputs,
        source_passages=source_passages,
    )
    merged = _merge_audit_results(
        llm_result=llm_result,
        heuristic_result=heuristic_result,
        original_output=step_output,
    )
    return merged


async def autofix_step(
    original_output: str,
    audit_result: AuditResult,
    sources: list[str],
    mistral: MistralClient,
) -> str:
    """Rewrite a flagged output using the issue list and source context."""

    source_blob = "\n\n".join(sources)
    corrected = await mistral.run_autofix(
        original=original_output,
        issues=audit_result.issues,
        sources=source_blob,
    )
    deterministic = _apply_issue_replacements(
        original=corrected or original_output,
        issues=audit_result.issues,
    )
    return deterministic.strip() or original_output


async def _gather_source_context(step_output: str, run_id: str) -> list[str]:
    """Retrieve a small, deduplicated source context bundle for auditing."""

    passages = await query_sources(step_output, run_id, n_results=5)
    for claim in _extract_claim_candidates(step_output)[:4]:
        passages.extend(await query_sources(claim, run_id, n_results=5))

    deduped: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        clean = passage.strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped[:10]


def _build_audit_prompt(
    step_output: str,
    user_goal: str,
    previous_outputs: list[str],
    source_passages: list[str],
) -> str:
    """Build the auditor prompt payload supplied to Mistral."""

    previous_text = "\n\n".join(previous_outputs) if previous_outputs else "None"
    source_text = "\n\n".join(source_passages) if source_passages else "None"
    return (
        f"{AUDITOR_SYSTEM_PROMPT.strip()}\n\n"
        f"Original user goal:\n{user_goal}\n\n"
        f"Current step output:\n{step_output}\n\n"
        f"Previous step outputs:\n{previous_text}\n\n"
        f"Retrieved source passages:\n{source_text}\n"
    )


def _run_rule_based_audit(
    step_output: str,
    user_goal: str,
    previous_outputs: list[str],
    source_passages: list[str],
) -> AuditResult:
    """Run deterministic checks to stabilize the audit outcome."""

    issues: list[AuditIssue] = []
    source_blob = "\n".join(source_passages)

    issues.extend(_detect_hallucinations(step_output, source_passages))
    issues.extend(_detect_contradictions(step_output, previous_outputs, source_passages))

    intent_issue = _detect_intent_drift(step_output, user_goal)
    if intent_issue is not None:
        issues.append(intent_issue)

    issues = _dedupe_issues(issues)
    verdict = "FLAG" if _should_flag(issues) else "PASS"
    drift_score = _score_issues(issues)
    summary = _build_summary(issues, source_blob)

    return AuditResult(
        step_id="",
        verdict=verdict,
        drift_score=drift_score,
        issues=issues,
        summary=summary,
        original_output=step_output,
        final_output=step_output,
        auto_fixed=False,
    )


def _extract_claim_candidates(text: str) -> list[str]:
    """Extract sentence-like claims worth grounding checks."""

    candidates: list[str] = []
    for sentence in _split_claim_sentences(text):
        if not sentence:
            continue
        if NUMERIC_PATTERN.search(sentence) or KEYWORD_PATTERN.search(sentence):
            candidates.append(sentence)
    return candidates


def _split_claim_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks without breaking decimal numbers."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(normalized):
        character = normalized[index]
        if character in ".!?":
            previous_char = normalized[index - 1] if index > 0 else ""
            next_char = normalized[index + 1] if index + 1 < len(normalized) else ""
            if character == "." and previous_char.isdigit() and next_char.isdigit():
                index += 1
                continue

            chunk = normalized[start : index + 1].strip()
            if chunk:
                sentences.append(chunk)
            start = index + 1
        index += 1

    tail = normalized[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _detect_hallucinations(step_output: str, source_passages: list[str]) -> list[AuditIssue]:
    """Flag numeric or factual claims that do not appear in retrieved sources."""

    issues: list[AuditIssue] = []
    source_numbers = {
        _normalize_numeric_token(number)
        for passage in source_passages
        for number in NUMERIC_PATTERN.findall(passage)
    }
    for claim in _extract_claim_candidates(step_output):
        numbers = NUMERIC_PATTERN.findall(claim)
        if not numbers:
            continue
        unsupported = [
            number
            for number in numbers
            if _normalize_numeric_token(number) not in source_numbers
        ]
        if not unsupported:
            continue

        supported = _find_supported_value(claim, source_passages, target_number=unsupported[0])
        if not supported:
            supported = _fallback_supported_value(
                source_passages=source_passages,
                target_number=unsupported[0],
                claim=claim,
            )
        actual_text = f" (actual: {supported})" if supported else ""
        suggestion = (
            f"Replace the unsupported figure with the sourced figure {supported}."
            if supported
            else "Remove the unsupported claim or qualify it as unverified."
        )
        issues.append(
            AuditIssue(
                type="HALLUCINATION",
                severity="HIGH",
                claim=unsupported[0],
                reason=(
                    "Claim "
                    f"'{_display_numeric_token(unsupported[0])}' does not appear "
                    f"in the retrieved sources{actual_text}."
                ),
                suggestion=suggestion,
            )
        )
    return issues


def _detect_contradictions(
    step_output: str, previous_outputs: list[str], source_passages: list[str]
) -> list[AuditIssue]:
    """Flag claims that contradict earlier step outputs on the same metric."""

    issues: list[AuditIssue] = []
    previous_claims = [
        claim
        for output in previous_outputs
        for claim in _extract_claim_candidates(output)
    ]
    if not previous_claims:
        return issues

    for claim in _extract_claim_candidates(step_output):
        numbers = NUMERIC_PATTERN.findall(claim)
        if not numbers:
            continue
        metric = _infer_metric(claim)
        if not metric:
            continue

        prior_claims_for_metric = [
            previous_claim
            for previous_claim in previous_claims
            if _infer_metric(previous_claim) == metric
            and _claims_share_entity_context(claim, previous_claim)
        ]
        if not prior_claims_for_metric:
            continue

        relevant_prior_claims = sorted(
            prior_claims_for_metric,
            key=lambda previous_claim: _claim_similarity(claim, previous_claim),
            reverse=True,
        )[:1]
        current_numbers = _numbers_for_metric(claim, metric) or numbers
        target_kind = _numeric_kind(current_numbers[0])
        current_number_keys = {
            _normalize_numeric_token(number) for number in current_numbers
        }
        sourced_value = _find_supported_value(
            claim,
            source_passages,
            target_number=current_numbers[0],
        )
        if sourced_value and _normalize_numeric_token(sourced_value) in current_number_keys:
            continue
        prior_numbers = [
            number
            for previous_claim in relevant_prior_claims
            for number in (_numbers_for_metric(previous_claim, metric) or NUMERIC_PATTERN.findall(previous_claim))
            if not target_kind or _numeric_kind(number) == target_kind
        ]
        conflicting = [
            number
            for number in prior_numbers
            if _normalize_numeric_token(number) not in current_number_keys
        ]
        if not conflicting:
            continue

        supported = _find_supported_value(claim, source_passages, target_number=numbers[0]) or conflicting[0]
        issues.append(
            AuditIssue(
                type="CONTRADICTION",
                severity="MEDIUM",
                claim=claim.strip(),
                reason=(
                    f"This {metric} claim conflicts with an earlier pipeline output "
                    f"that used {conflicting[0]}."
                ),
                suggestion=f"Use the consistent, sourced {metric} value {supported}.",
            )
        )
    return issues


def _detect_intent_drift(step_output: str, user_goal: str) -> AuditIssue | None:
    """Flag outputs that stop addressing the original user goal."""

    goal_tokens = {token for token in re.findall(r"\w+", user_goal.lower()) if len(token) > 3}
    output_tokens = {token for token in re.findall(r"\w+", step_output.lower()) if len(token) > 3}
    if not goal_tokens or not output_tokens:
        return None

    overlap = len(goal_tokens & output_tokens) / len(goal_tokens)
    if overlap >= 0.15 or len(step_output.split()) < 35:
        return None

    return AuditIssue(
        type="INTENT_DRIFT",
        severity="LOW",
        claim=step_output[:160].strip(),
        reason="The output does not substantially address the original user goal.",
        suggestion="Refocus the answer on the stated objective and remove unrelated detail.",
    )


def _merge_audit_results(
    llm_result: AuditResult,
    heuristic_result: AuditResult,
    original_output: str,
) -> AuditResult:
    """Combine model and rule-based findings into a final step verdict."""

    issues = list(heuristic_result.issues)
    if issues:
        drift_score = max(heuristic_result.drift_score, _score_issues(issues))
    else:
        issues = _dedupe_issues(
            [issue for issue in llm_result.issues if issue.severity == "HIGH"]
        )
        drift_score = max(llm_result.drift_score if issues else 0.0, _score_issues(issues))

    verdict = "FLAG" if _should_flag(issues) or drift_score > 0.3 else "PASS"
    summary = heuristic_result.summary
    if verdict == "PASS" and not issues and llm_result.summary:
        summary = llm_result.summary
    elif verdict == "FLAG" and llm_result.summary and not heuristic_result.issues:
        summary = llm_result.summary

    return AuditResult(
        step_id="",
        verdict=verdict,
        drift_score=max(drift_score, _score_issues(issues)),
        issues=issues,
        summary=summary,
        original_output=original_output,
        final_output=original_output,
        auto_fixed=False,
    )


def _dedupe_issues(issues: list[AuditIssue]) -> list[AuditIssue]:
    """Remove duplicate issues while preserving order."""

    deduped: list[AuditIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        key = (issue.type, issue.claim.strip(), issue.reason.strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _should_flag(issues: list[AuditIssue]) -> bool:
    """Apply the DriftWatch flagging threshold."""

    return any(issue.severity == "HIGH" for issue in issues)


def _score_issues(issues: list[AuditIssue]) -> float:
    """Convert issue severities into a normalized drift score."""

    if not issues:
        return 0.0
    weights = {"LOW": 0.15, "MEDIUM": 0.35, "HIGH": 0.75}
    return min(1.0, max(weights[issue.severity] for issue in issues))


def _build_summary(issues: list[AuditIssue], source_blob: str) -> str:
    """Create a plain-English summary for the frontend."""

    if not issues:
        return "Output is aligned with the goal and grounded in the retrieved sources."

    primary = issues[0]
    if primary.type == "HALLUCINATION":
        actual = _extract_suggested_value(primary.suggestion)
        actual_suffix = f" (actual: {actual})" if actual else ""
        return (
            "Hallucination detected: "
            f"{_display_numeric_token(primary.claim)} not found in sources{actual_suffix}."
        )
    if primary.type == "CONTRADICTION":
        return "Contradiction detected between this step and an earlier pipeline output."
    if primary.type == "INTENT_DRIFT":
        return "Intent drift detected: the output veered away from the original goal."
    return "Audit flagged one or more issues."


def _find_supported_value(
    claim: str, source_passages: list[str], target_number: str | None = None
) -> str | None:
    """Find the most plausible sourced replacement value for a claim."""

    metric = _infer_metric(claim)
    scored_matches: list[tuple[float, str]] = []
    target_kind = _numeric_kind(target_number) if target_number else ""
    claim_entities = _extract_entities(claim)
    for passage in source_passages:
        passage_metric = _infer_metric(passage)
        if metric and passage_metric and metric != passage_metric:
            continue
        for number in NUMERIC_PATTERN.findall(passage):
            number_kind = _numeric_kind(number)
            if target_kind and number_kind and target_kind != number_kind:
                continue
            local_context = _local_number_context(passage, number)
            if claim_entities and not (claim_entities & _extract_entities(local_context)):
                local_entities = _extract_entities(local_context)
                if any(entity.lower().endswith("corp") for entity in local_entities):
                    continue
            entity_score = _entity_alignment_score(claim_entities, local_context)
            similarity = SequenceMatcher(None, claim.lower(), local_context.lower()).ratio()
            if metric and passage_metric == metric:
                similarity += 0.2
            if target_kind and number_kind == target_kind:
                similarity += 0.3
            similarity += entity_score
            scored_matches.append((similarity, number))
    if not scored_matches:
        return None
    scored_matches.sort(reverse=True)
    return _display_numeric_token(scored_matches[0][1])


def _infer_metric(text: str) -> str:
    """Infer a coarse business metric label from a sentence."""

    match = KEYWORD_PATTERN.search(text)
    return match.group(1).lower() if match else ""


def _extract_suggested_value(suggestion: str) -> str | None:
    """Extract a numeric replacement from an issue suggestion string."""

    match = NUMERIC_PATTERN.search(suggestion)
    return _display_numeric_token(match.group(0)) if match else None


def _numeric_kind(value: str | None) -> str:
    """Classify a numeric token so replacements preserve the metric shape."""

    if not value:
        return ""
    normalized = _normalize_numeric_token(value)
    if "%" in normalized:
        return "percent"
    if normalized.startswith("$") or normalized.endswith(("b", "m")):
        return "money"
    if normalized.endswith("x"):
        return "multiple"
    if re.fullmatch(r"(19|20)\d{2}", normalized):
        return "year"
    return "number"


def _expected_numeric_kind(metric: str) -> str:
    """Map a coarse metric label to the numeric shape it usually carries."""

    normalized = metric.lower()
    if normalized in {"revenue", "earnings", "price target"}:
        return "money"
    if normalized in {"market share", "inflation", "growth", "r&d"}:
        return "percent"
    if normalized == "p/e":
        return "multiple"
    return ""


def _numbers_for_metric(text: str, metric: str) -> list[str]:
    """Find the most relevant numeric tokens near a metric label."""

    metric_pattern = re.compile(re.escape(metric), re.IGNORECASE)
    metric_matches = list(metric_pattern.finditer(text))
    number_matches = list(NUMERIC_PATTERN.finditer(text))
    if not metric_matches or not number_matches:
        return []

    expected_kind = _expected_numeric_kind(metric)
    selected: list[str] = []
    for metric_match in metric_matches:
        candidates = [
            match
            for match in number_matches
            if not expected_kind or _numeric_kind(match.group(0)) == expected_kind
        ] or number_matches
        nearest = min(
            candidates,
            key=lambda match: (
                abs(match.start() - metric_match.start()),
                -(match.end() - match.start()),
            ),
        )
        if abs(nearest.start() - metric_match.start()) > 48:
            continue
        value = nearest.group(0)
        if value not in selected:
            selected.append(value)
    return selected


def _apply_issue_replacements(original: str, issues: list[AuditIssue]) -> str:
    """Apply simple in-place replacements from issue suggestions."""

    corrected = original
    for issue in issues:
        replacement = _extract_suggested_value(issue.suggestion)
        if issue.type == "HALLUCINATION" and replacement and issue.claim:
            corrected = corrected.replace(issue.claim, replacement)
    return corrected


def _extract_entities(text: str) -> set[str]:
    """Extract coarse named entities used to keep replacements on the same subject."""

    return {
        token
        for token in re.findall(r"\b[A-Z][A-Za-z0-9]+\b", text)
        if len(token) > 2
    }


def _claims_share_entity_context(claim: str, previous_claim: str) -> bool:
    """Require corp-like entities to overlap before treating claims as contradictory."""

    claim_entities = {
        entity.lower()
        for entity in _extract_entities(claim)
        if entity.lower().endswith("corp")
    }
    previous_entities = {
        entity.lower()
        for entity in _extract_entities(previous_claim)
        if entity.lower().endswith("corp")
    }
    if claim_entities or previous_entities:
        return bool(claim_entities & previous_entities)
    return True


def _claim_similarity(claim: str, previous_claim: str) -> float:
    """Score which prior claim is the best comparison candidate."""

    similarity = SequenceMatcher(None, claim.lower(), previous_claim.lower()).ratio()
    claim_entities = _extract_entities(claim)
    previous_entities = _extract_entities(previous_claim)
    overlap = len(claim_entities & previous_entities)
    return similarity + (0.15 * overlap)


def _entity_alignment_score(claim_entities: set[str], passage: str) -> float:
    """Score how well a source passage matches the named entities in the claim."""

    if not claim_entities:
        return 0.0

    passage_entities = _extract_entities(passage)
    score = 0.0
    for entity in claim_entities & passage_entities:
        score += 0.15
    extra_corp_entities = {
        entity
        for entity in passage_entities - claim_entities
        if entity.lower().endswith("corp")
    }
    if extra_corp_entities:
        score -= 0.2
    return score


def _local_number_context(passage: str, number: str) -> str:
    """Return the smallest sentence-like segment containing a candidate number."""

    for sentence in _split_claim_sentences(passage):
        if number in sentence:
            return sentence
    return passage


def _fallback_supported_value(
    source_passages: list[str], target_number: str, claim: str
) -> str | None:
    """Pick the first source value with the same numeric type and subject."""

    target_kind = _numeric_kind(target_number)
    claim_entities = _extract_entities(claim)
    for passage in source_passages:
        local_entities = _extract_entities(passage)
        if claim_entities and not (claim_entities & local_entities):
            continue
        for number in NUMERIC_PATTERN.findall(passage):
            if not target_kind or _numeric_kind(number) == target_kind:
                return _display_numeric_token(number)
    for passage in source_passages:
        for number in NUMERIC_PATTERN.findall(passage):
            if not target_kind or _numeric_kind(number) == target_kind:
                return _display_numeric_token(number)
    return None


def _normalize_numeric_token(value: str) -> str:
    """Normalize numeric tokens so equivalent forms compare cleanly."""

    normalized = value.strip().lower().replace(",", "")
    normalized = normalized.replace("billion", "b")
    normalized = normalized.replace("million", "m")
    normalized = normalized.replace("percent", "%")
    normalized = normalized.replace("times", "x")
    normalized = normalized.replace(" ", "")
    normalized = re.sub(r"^\+", "", normalized)
    return normalized


def _display_numeric_token(value: str) -> str:
    """Render numeric tokens in a compact human-readable form."""

    normalized = _normalize_numeric_token(value)
    if normalized.endswith("b"):
        return f"{normalized[:-1]}B"
    if normalized.endswith("m"):
        return f"{normalized[:-1]}M"
    return normalized


def _issue_overlaps(candidate: AuditIssue, existing: list[AuditIssue]) -> bool:
    """Check whether an incoming issue duplicates an existing finding."""

    candidate_claim = _normalize_numeric_token(candidate.claim)
    for issue in existing:
        if issue.type != candidate.type:
            continue
        if candidate_claim and candidate_claim == _normalize_numeric_token(issue.claim):
            return True
        if candidate.claim.strip() == issue.claim.strip():
            return True
    return False
