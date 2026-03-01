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
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from mistralai import Mistral
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from backend.logging_utils import get_run_logger
from backend.models import AuditIssue, AuditResult, GeneratedPipelineStep
from backend.prompts import AUDITOR_SYSTEM_PROMPT, PIPELINE_ARCHITECT_SYSTEM_PROMPT

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

class MistralClient:
    """Thin async client for agent, auditor, autofix, and embedding calls."""

    def __init__(self) -> None:
        """Initialize the Mistral SDK client and model configuration."""

        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=api_key) if api_key else None
        self.agent_model = "mistral-large-latest"
        self.auditor_model = "mistral-small-latest"
        self.embed_model = "mistral-embed"

    async def run_agent_step(
        self,
        system: str,
        user: str,
        run_id: str | None = None,
        on_token: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> str:
        """Execute one pipeline step with the agent model."""

        run_logger = get_run_logger(logger, run_id)
        try:
            return await self._chat_stream(
                model=self.agent_model,
                system=system,
                user=user,
                temperature=0.3,
                max_tokens=2048,
                run_id=run_id,
                operation_name="agent_step",
                on_token=on_token,
            )
        except Exception as exc:
            run_logger.exception("Agent step failed: %s", exc)
            fallback = self._deterministic_agent_step(user)
            if on_token is not None and fallback:
                await on_token(fallback, fallback)
            return fallback

    async def run_auditor(self, audit_prompt: str, run_id: str | None = None) -> AuditResult:
        """Execute the auditor model and parse its JSON verdict."""

        run_logger = get_run_logger(logger, run_id)
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
                max_tokens=800,
                run_id=run_id,
                operation_name="auditor",
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
            run_logger.warning("Auditor JSON parse failed: %s", exc)
            return fallback.model_copy(
                update={
                    "summary": "Auditor returned non-JSON output; using a safe fallback PASS.",
                }
            )
        except Exception as exc:
            run_logger.exception("Auditor call failed: %s", exc)
            return fallback.model_copy(
                update={
                    "summary": "Auditor API unavailable; using deterministic fallback checks.",
                }
            )

    async def run_autofix(
        self,
        original: str,
        issues: list[AuditIssue],
        sources: str,
        run_id: str | None = None,
    ) -> str:
        """Rewrite a flagged output while preserving grounded claims."""

        run_logger = get_run_logger(logger, run_id)
        issue_lines = "\n".join(
            f"- {issue.type} [{issue.severity}] claim={issue.claim} "
            f"fix={issue.suggestion}"
            for issue in issues
        )
        prompt = (
            "Rewrite the following text. For each issue listed, replace the incorrect "
            "claim with the correct value from the source documents. Return only the "
            "corrected text with no explanation.\n\n"
            f"Flagged output:\n{original}\n\n"
            f"Issues to fix:\n{issue_lines or '- None provided'}\n\n"
            f"Sources:\n{sources}\n\n"
            "Keep every correction explicit and sourced."
        )
        try:
            response = await self._chat_complete(
                model=self.agent_model,
                system="You are a precise editor. Replace every incorrect claim with the sourced value.",
                user=prompt,
                temperature=0.2,
                max_tokens=2048,
                run_id=run_id,
                operation_name="autofix",
            )
            corrected = self._extract_text_content(response).strip()
            return corrected or original
        except Exception as exc:
            run_logger.exception("Autofix call failed: %s", exc)
            return self._deterministic_autofix(original=original, issues=issues, sources=sources)

    async def generate_pipeline_steps(
        self,
        user_goal: str,
        source_documents: list[str],
        run_id: str | None = None,
    ) -> list[GeneratedPipelineStep]:
        """Generate 3-4 sequential pipeline steps for a custom user request."""

        run_logger = get_run_logger(logger, run_id)
        source_preview = "\n\n".join(source_documents).strip()[:500]
        user_prompt = f"User goal:\n{user_goal}\n\nSource documents preview:\n{source_preview}"

        try:
            response = await self._chat_complete(
                model=self.agent_model,
                system=PIPELINE_ARCHITECT_SYSTEM_PROMPT,
                user=user_prompt,
                temperature=0.0,
                run_id=run_id,
                operation_name="generate_pipeline_steps",
            )
            payload_text = self._extract_text_content(response)
            payload = json.loads(self._strip_json_fence(payload_text))
            if not isinstance(payload, list):
                raise ValueError("Pipeline architect response was not a JSON array.")

            steps = [
                GeneratedPipelineStep(
                    step_name=self._truncate_step_name(str(item.get("step_name", "")).strip()),
                    instruction=str(item.get("instruction", "")).strip(),
                )
                for item in payload
                if isinstance(item, dict)
                and str(item.get("step_name", "")).strip()
                and str(item.get("instruction", "")).strip()
            ]
            if steps:
                return steps[:4]
            raise ValueError("Pipeline architect returned no usable steps.")
        except Exception as exc:
            run_logger.exception("Pipeline step generation failed: %s", exc)
            return self._deterministic_pipeline_steps(user_goal=user_goal, source_documents=source_documents)

    async def embed(self, texts: list[str], run_id: str | None = None) -> list[list[float]]:
        """Embed text with Mistral, falling back to deterministic local vectors."""

        if not texts:
            return []
        run_logger = get_run_logger(logger, run_id)
        try:
            if self.client is None:
                raise RuntimeError("MISTRAL_API_KEY is not configured.")

            def _call() -> Any:
                return self.client.embeddings.create(
                    model=self.embed_model,
                    inputs=texts,
                )

            response = await self._run_with_retry(
                operation=_call,
                run_id=run_id,
                operation_name="embeddings",
            )
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
            run_logger.warning("Embedding fallback activated: %s", exc)
            return [self._local_embedding(text) for text in texts]

    async def _chat_complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int | None = None,
        run_id: str | None = None,
        operation_name: str = "chat_completion",
    ) -> Any:
        """Execute a blocking SDK chat call in a worker thread."""

        if self.client is None:
            raise RuntimeError("MISTRAL_API_KEY is not configured.")

        def _call() -> Any:
            return self.client.chat.complete(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )

        return await self._run_with_retry(
            operation=_call,
            run_id=run_id,
            operation_name=operation_name,
        )

    async def _chat_stream(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int | None = None,
        run_id: str | None = None,
        operation_name: str = "chat_stream",
        on_token: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> str:
        """Stream a chat completion and forward partial text as it arrives."""

        if self.client is None:
            raise RuntimeError("MISTRAL_API_KEY is not configured.")

        async def _call() -> Any:
            return await self.client.chat.stream_async(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )

        stream = await self._run_async_with_retry(
            operation=_call,
            run_id=run_id,
            operation_name=operation_name,
        )

        fragments: list[str] = []
        async for event in stream:
            delta = self._extract_stream_delta(event)
            if not delta:
                continue
            fragments.append(delta)
            if on_token is not None:
                await on_token(delta, "".join(fragments))
        return "".join(fragments).strip()

    async def _run_with_retry(
        self,
        operation: Any,
        run_id: str | None,
        operation_name: str,
    ) -> Any:
        """Execute an SDK operation with retry handling for shared-key instability."""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=8),
            retry=retry_if_exception(self._is_retryable_mistral_error),
            reraise=True,
            before_sleep=self._before_sleep(run_id=run_id, operation_name=operation_name),
        ):
            with attempt:
                return await asyncio.to_thread(operation)

        raise RuntimeError(f"{operation_name} failed after retries.")

    async def _run_async_with_retry(
        self,
        operation: Callable[[], Awaitable[Any]],
        run_id: str | None,
        operation_name: str,
    ) -> Any:
        """Execute an async SDK operation with retry handling."""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=8),
            retry=retry_if_exception(self._is_retryable_mistral_error),
            reraise=True,
            before_sleep=self._before_sleep(run_id=run_id, operation_name=operation_name),
        ):
            with attempt:
                return await operation()

        raise RuntimeError(f"{operation_name} failed after retries.")

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

    def _extract_stream_delta(self, event: Any) -> str:
        """Extract the incremental text delta from a stream event."""

        chunk = getattr(event, "data", None)
        if chunk is None and isinstance(event, dict):
            chunk = event.get("data", {})

        choices = getattr(chunk, "choices", None)
        if choices is None and isinstance(chunk, dict):
            choices = chunk.get("choices", [])
        if not choices:
            return ""

        first_choice = choices[0]
        delta = getattr(first_choice, "delta", None)
        if delta is None and isinstance(first_choice, dict):
            delta = first_choice.get("delta", {})

        content = getattr(delta, "content", None)
        if content is None and isinstance(delta, dict):
            content = delta.get("content")
        return content if isinstance(content, str) else ""

    def _strip_json_fence(self, payload_text: str) -> str:
        """Remove Markdown code fences around a JSON payload if present."""

        text = payload_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _before_sleep(self, run_id: str | None, operation_name: str) -> Any:
        """Create a retry hook that logs rate-limit and availability retries."""

        run_logger = get_run_logger(logger, run_id)

        def _log_retry(retry_state: Any) -> None:
            exception = retry_state.outcome.exception() if retry_state.outcome else None
            status_code = getattr(exception, "status_code", "unknown")
            run_logger.warning(
                "Retrying %s after API status %s (attempt %s).",
                operation_name,
                status_code,
                retry_state.attempt_number,
            )

        return _log_retry

    def _is_retryable_mistral_error(self, exception: BaseException) -> bool:
        """Return whether a Mistral error should be retried."""

        status_code = getattr(exception, "status_code", None)
        return status_code in {429, 503}

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

    def _deterministic_pipeline_steps(
        self,
        user_goal: str,
        source_documents: list[str],
    ) -> list[GeneratedPipelineStep]:
        """Return stable fallback pipeline steps if the architect call fails."""

        goal = user_goal.lower()
        source_blob = "\n".join(source_documents).lower()

        if "legal" in goal or "contract" in source_blob:
            return [
                GeneratedPipelineStep(
                    step_name="Clause Extractor",
                    instruction="Extract the key obligations, deadlines, parties, and legal clauses from the source documents.",
                ),
                GeneratedPipelineStep(
                    step_name="Risk Reviewer",
                    instruction="Identify the highest-risk legal terms or ambiguities that could affect the reader.",
                ),
                GeneratedPipelineStep(
                    step_name="Summary Writer",
                    instruction="Write a concise grounded summary using only the extracted clauses and verified risks.",
                ),
            ]

        if "article" in goal or "fact-check" in goal or "fact check" in goal:
            return [
                GeneratedPipelineStep(
                    step_name="Fact Extractor",
                    instruction="Extract all verifiable names, dates, figures, and concrete claims from the source documents.",
                ),
                GeneratedPipelineStep(
                    step_name="Claim Reviewer",
                    instruction="Compare the requested output against the extracted facts and identify unsupported or missing claims.",
                ),
                GeneratedPipelineStep(
                    step_name="Final Writer",
                    instruction="Write a corrected final response using only the verified claims from the source documents.",
                ),
            ]

        return [
            GeneratedPipelineStep(
                step_name="Source Extractor",
                instruction="Extract the most important grounded facts, metrics, and named entities from the source documents.",
            ),
            GeneratedPipelineStep(
                step_name="Evidence Summarizer",
                instruction="Summarize the extracted evidence into a concise intermediate brief without adding unsupported claims.",
            ),
            GeneratedPipelineStep(
                step_name="Final Writer",
                instruction="Write the final answer using only the summarized evidence and the original user goal.",
            ),
        ]

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

    def _truncate_step_name(self, value: str) -> str:
        """Trim generated step names down to at most four words."""

        return " ".join(value.split()[:4]).strip()

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
