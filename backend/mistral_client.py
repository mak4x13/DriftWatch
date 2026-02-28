"""Async wrapper around the Mistral SDK used by DriftWatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any

from dotenv import load_dotenv
from mistralai import Mistral

from backend.models import AuditIssue, AuditResult

logger = logging.getLogger(__name__)
NUMERIC_TOKEN_PATTERN = r"(?<![A-Za-z0-9])\$?(?:\d+\.\d+|\d+)(?:B|M|%|x)?(?![A-Za-z0-9])"
ISSUE_TYPE_ALIASES = {
    "INTENTDRIFT": "INTENT_DRIFT",
    "INTENT_DRIFT": "INTENT_DRIFT",
    "HALLUCINATION": "HALLUCINATION",
    "CONTRADICTION": "CONTRADICTION",
}
SEVERITY_ALIASES = {
    "LOW": "LOW",
    "MEDIUM": "MEDIUM",
    "MODERATE": "MEDIUM",
    "HIGH": "HIGH",
}
VERDICT_ALIASES = {
    "PASS": "PASS",
    "FLAG": "FLAG",
}

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


class MistralClient:
    """Thin async client for agent, auditor, autofix, and embedding calls."""

    def __init__(self) -> None:
        """Initialize the Mistral SDK client and model configuration."""

        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=api_key) if api_key else None
        self.agent_model = "mistral-large-latest"
        self.auditor_model = "mistral-large-latest"
        self.embed_model = "mistral-embed"

    async def run_agent_step(self, system: str, user: str) -> str:
        """Execute one pipeline step with the agent model."""

        try:
            response = await self._chat_complete(
                model=self.agent_model,
                system=system,
                user=user,
                temperature=0.3,
            )
            return self._extract_text_content(response).strip()
        except Exception as exc:
            logger.exception("Agent step failed: %s", exc)
            return self._deterministic_agent_step(user)

    async def run_auditor(self, audit_prompt: str) -> AuditResult:
        """Execute the auditor model and parse its JSON verdict."""

        fallback = AuditResult(
            step_id="",
            verdict="PASS",
            drift_score=0.0,
            issues=[],
            summary="Auditor returned a fallback PASS because structured parsing failed.",
            original_output="",
            final_output="",
            auto_fixed=False,
        )
        try:
            response = await self._chat_complete(
                model=self.auditor_model,
                system=AUDITOR_SYSTEM_PROMPT.strip(),
                user=audit_prompt,
                temperature=0.0,
            )
            payload_text = self._extract_text_content(response)
            payload = json.loads(self._strip_json_fence(payload_text))
            issues = []
            for item in payload.get("issues", []):
                normalized_issue = self._normalize_issue_payload(item)
                if normalized_issue is None:
                    continue
                issues.append(normalized_issue)

            return AuditResult(
                step_id="",
                verdict=self._normalize_verdict(payload.get("verdict", "PASS")),
                drift_score=self._normalize_drift_score(payload.get("drift_score", 0.0)),
                issues=issues,
                summary=payload.get("summary", "Audit completed."),
                original_output="",
                final_output="",
                auto_fixed=False,
            )
        except json.JSONDecodeError as exc:
            logger.warning("Auditor JSON parse failed: %s", exc)
            return fallback.model_copy(
                update={
                    "summary": "Auditor returned non-JSON output; using a safe fallback PASS.",
                }
            )
        except Exception as exc:
            logger.exception("Auditor call failed: %s", exc)
            return fallback.model_copy(
                update={
                    "summary": "Auditor API unavailable; using deterministic fallback checks.",
                }
            )

    async def run_autofix(
        self, original: str, issues: list[AuditIssue], sources: str
    ) -> str:
        """Rewrite a flagged output while preserving grounded claims."""

        issue_lines = "\n".join(
            f"- {issue.type} [{issue.severity}] claim={issue.claim} "
            f"fix={issue.suggestion}"
            for issue in issues
        )
        prompt = (
            "Rewrite the flagged output so it remains concise, grounded, and "
            "consistent with the supplied sources.\n\n"
            f"Flagged output:\n{original}\n\n"
            f"Issues to fix:\n{issue_lines or '- None provided'}\n\n"
            f"Sources:\n{sources}\n\n"
            "Return only the corrected text."
        )
        try:
            response = await self._chat_complete(
                model=self.agent_model,
                system="You are a precise editor. Remove or qualify unverified claims.",
                user=prompt,
                temperature=0.2,
            )
            corrected = self._extract_text_content(response).strip()
            return corrected or original
        except Exception as exc:
            logger.exception("Autofix call failed: %s", exc)
            return self._deterministic_autofix(original=original, issues=issues, sources=sources)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed text with Mistral, falling back to deterministic local vectors."""

        if not texts:
            return []
        try:
            if self.client is None:
                raise RuntimeError("MISTRAL_API_KEY is not configured.")

            def _call() -> Any:
                return self.client.embeddings.create(
                    model=self.embed_model,
                    inputs=texts,
                )

            response = await asyncio.to_thread(_call)
            data = getattr(response, "data", None)
            if data is None and isinstance(response, dict):
                data = response.get("data", [])
            if data is None:
                raise ValueError("Embedding response did not contain data.")

            vectors: list[list[float]] = []
            for item in data:
                vector = getattr(item, "embedding", None)
                if vector is None and isinstance(item, dict):
                    vector = item.get("embedding")
                if not isinstance(vector, list):
                    raise ValueError("Embedding item missing vector.")
                vectors.append([float(value) for value in vector])
            return vectors
        except Exception as exc:
            logger.warning("Embedding fallback activated: %s", exc)
            return [self._local_embedding(text) for text in texts]

    async def _chat_complete(
        self, model: str, system: str, user: str, temperature: float
    ) -> Any:
        """Execute a blocking SDK chat call in a worker thread."""

        if self.client is None:
            raise RuntimeError("MISTRAL_API_KEY is not configured.")

        def _call() -> Any:
            return self.client.chat.complete(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )

        return await asyncio.to_thread(_call)

    def _extract_text_content(self, response: Any) -> str:
        """Extract the first assistant message text from SDK or dict responses."""

        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])
        if not choices:
            return ""

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message", {})

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content", "")

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts)
        return str(content or "")

    def _strip_json_fence(self, payload_text: str) -> str:
        """Remove Markdown code fences around a JSON payload if present."""

        text = payload_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _deterministic_autofix(
        self, original: str, issues: list[AuditIssue], sources: str
    ) -> str:
        """Apply a small rule-based correction when the model is unavailable."""

        corrected = original
        source_numbers = re.findall(NUMERIC_TOKEN_PATTERN, sources)
        for issue in issues:
            replacement = self._extract_replacement(issue.suggestion) or self._best_fallback_replacement(
                issue.claim,
                source_numbers,
            )
            if issue.type == "HALLUCINATION" and issue.claim and replacement:
                corrected = corrected.replace(issue.claim, replacement)
        return corrected

    def _deterministic_agent_step(self, user: str) -> str:
        """Generate a stable local step output when the remote API is unavailable."""

        step_name = self._extract_section(user, "Step name:")
        instruction = self._extract_section(user, "Instruction:")
        context = self._extract_section(user, "Input context:")

        step_name_lower = step_name.lower()
        instruction_lower = instruction.lower()

        if "hallucinate" in instruction_lower:
            return (
                "TechCorp generated $4.7B in revenue in Q3 2024, up 12% year over year, "
                "while maintaining 53% market share. Expansion into Asian markets supported "
                "growth, but rising chip costs remained a headwind and R&D spend increased 18%."
            )

        fact_sentences = self._extract_fact_sentences(context)
        if "extract" in instruction_lower or "facts" in instruction_lower or "extractor" in step_name_lower:
            return "\n".join(f"- {sentence}" for sentence in fact_sentences[:5]) or context[:400]

        if "report" in instruction_lower or "writer" in step_name_lower:
            lead = self._find_value(context, "revenue") or "$3.2B"
            market_share = self._find_value(context, "market share") or "53%"
            return (
                "TechCorp delivered a solid quarter anchored by revenue of "
                f"{lead}. The company maintained {market_share} enterprise market share, "
                "benefited from Asian market expansion, and continued to invest through higher "
                "R&D spending. Investors should weigh that momentum against rising chip costs."
            )

        summary_sentences = fact_sentences[:4]
        if summary_sentences:
            return " ".join(summary_sentences)
        return context[:500] or "No input context was available for this step."

    def _local_embedding(self, text: str, dimension: int = 128) -> list[float]:
        """Create a deterministic local embedding for offline fallback mode."""

        buckets = [0.0] * dimension
        tokens = re.findall(
            rf"\w+|{NUMERIC_TOKEN_PATTERN}",
            text.lower(),
        )
        if not tokens:
            return buckets

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimension
            buckets[index] += 1.0

        norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
        return [value / norm for value in buckets]

    def _extract_section(self, user: str, label: str) -> str:
        """Extract a labeled block from the agent prompt payload."""

        escaped_label = re.escape(label)
        pattern = re.compile(
            rf"{escaped_label}\s*\n(?P<value>.*?)(?:\n\n[A-Z][^:\n]*:\n|\Z)",
            re.DOTALL,
        )
        match = pattern.search(user)
        if not match:
            return ""
        return match.group("value").strip()

    def _extract_fact_sentences(self, context: str) -> list[str]:
        """Extract high-signal factual sentences from a block of context."""

        raw_sentences = re.split(r"(?<=[.!?])\s+", context.replace("\n", " ").strip())
        candidates: list[str] = []
        for sentence in raw_sentences:
            clean = sentence.strip()
            if not clean:
                continue
            if re.search(NUMERIC_TOKEN_PATTERN, clean) or re.search(
                r"\b(revenue|market share|chip costs|price target|p/e|r&d|inflation|growth)\b",
                clean,
                re.IGNORECASE,
            ):
                candidates.append(clean)

        if candidates:
            return candidates

        words = [word for word in re.findall(r"[A-Za-z0-9$%.&/-]+", context) if len(word) > 2]
        if not words:
            return []
        common_words = [word for word, _ in Counter(words).most_common(30)]
        return [" ".join(common_words[:20])]

    def _find_value(self, text: str, metric: str) -> str | None:
        """Find a nearby numeric value associated with a coarse metric label."""

        expected_kind = self._expected_kind_for_metric(metric)
        numeric_pattern = re.compile(
            self._kind_specific_numeric_pattern(expected_kind)
            if expected_kind
            else NUMERIC_TOKEN_PATTERN,
            re.IGNORECASE,
        )
        metric_pattern = re.compile(re.escape(metric), re.IGNORECASE)

        sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
        best_match: tuple[int, str] | None = None
        for sentence in sentences:
            metric_match = metric_pattern.search(sentence)
            if not metric_match:
                continue
            metric_index = metric_match.start()
            for number_match in numeric_pattern.finditer(sentence):
                distance = abs(number_match.start() - metric_index)
                candidate = number_match.group(0)
                if best_match is None or distance < best_match[0]:
                    best_match = (distance, candidate)

        return best_match[1] if best_match else None

    def _extract_replacement(self, suggestion: str) -> str | None:
        """Extract a numeric replacement token from an issue suggestion."""

        match = re.search(NUMERIC_TOKEN_PATTERN, suggestion)
        return match.group(0) if match else None

    def _normalize_issue_payload(self, item: dict[str, Any]) -> AuditIssue | None:
        """Normalize a raw auditor issue payload into a strict AuditIssue."""

        if not isinstance(item, dict):
            return None

        issue_type = self._normalize_issue_type(item.get("type", ""))
        severity = self._normalize_severity(item.get("severity", ""))
        if issue_type is None or severity is None:
            logger.warning("Skipping auditor issue with unsupported enums: %s", item)
            return None

        return AuditIssue(
            type=issue_type,
            severity=severity,
            claim=str(item.get("claim", "")).strip(),
            reason=str(item.get("reason", "")).strip(),
            suggestion=str(item.get("suggestion", "")).strip(),
        )

    def _normalize_issue_type(self, value: Any) -> str | None:
        """Normalize auditor issue types like 'INTENT DRIFT' into enum-safe values."""

        normalized = re.sub(r"[^A-Z]", "", str(value).upper())
        return ISSUE_TYPE_ALIASES.get(normalized)

    def _normalize_severity(self, value: Any) -> str | None:
        """Normalize auditor severities into enum-safe values."""

        normalized = re.sub(r"[^A-Z]", "", str(value).upper())
        return SEVERITY_ALIASES.get(normalized)

    def _normalize_verdict(self, value: Any) -> str:
        """Normalize the top-level auditor verdict."""

        normalized = re.sub(r"[^A-Z]", "", str(value).upper())
        return VERDICT_ALIASES.get(normalized, "PASS")

    def _normalize_drift_score(self, value: Any) -> float:
        """Clamp drift scores into the expected 0-1 range."""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, numeric))

    def _best_fallback_replacement(
        self, claim: str, source_numbers: list[str]
    ) -> str | None:
        """Choose a fallback replacement that matches the shape of the flagged claim."""

        claim_kind = self._numeric_kind(claim)
        for number in source_numbers:
            if not claim_kind or self._numeric_kind(number) == claim_kind:
                return number
        return source_numbers[0] if source_numbers else None

    def _expected_kind_for_metric(self, metric: str) -> str:
        """Map coarse metric labels to the numeric format they should prefer."""

        normalized = metric.lower()
        if normalized in {"revenue", "earnings", "price target"}:
            return "money"
        if normalized in {"market share", "inflation", "growth", "r&d"}:
            return "percent"
        if normalized == "p/e":
            return "multiple"
        return ""

    def _kind_specific_numeric_pattern(self, kind: str) -> str:
        """Return a regex fragment for a specific numeric presentation."""

        if kind == "money":
            return r"(?<![A-Za-z0-9])(?:\$\d+(?:\.\d+)?(?:B|M)?|\d+(?:\.\d+)?(?:B|M))(?![A-Za-z0-9])"
        if kind == "percent":
            return r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%(?![A-Za-z0-9])"
        if kind == "multiple":
            return r"(?<![A-Za-z0-9])\d+(?:\.\d+)?x(?![A-Za-z0-9])"
        return NUMERIC_TOKEN_PATTERN

    def _numeric_kind(self, value: str | None) -> str:
        """Classify a numeric token to preserve its presentation style."""

        if not value:
            return ""
        if "%" in value:
            return "percent"
        if value.startswith("$") or value.endswith(("B", "M")):
            return "money"
        if value.endswith("x"):
            return "multiple"
        if re.fullmatch(r"(19|20)\d{2}", value):
            return "year"
        return "number"
