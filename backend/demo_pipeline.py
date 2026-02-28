"""Prebuilt hackathon demo pipeline for DriftWatch."""

from __future__ import annotations

from backend.models import PipelineRequest, PipelineStep

DEMO_SOURCE_DOCUMENTS = [
    """TechCorp Q3 2024 Earnings: Revenue reached $3.2B, up 12% YoY.
       CEO stated: expansion into Asian markets drove growth.""",
    """Market Analysis: TechCorp faces headwinds from rising chip costs.
       Analyst consensus price target: $142. Current P/E ratio: 24x.""",
    """Competitor Report: RivalCorp posted $2.8B revenue same quarter.
       TechCorp maintains 53% market share in enterprise segment.""",
    """TechCorp Product Update: New AI chip released Q3, volume sales
       expected Q1 2026. R&D spend increased 18% this quarter.""",
    """Economic Context: Fed rate unchanged. Tech sector up 4% this
       quarter. Inflation at 2.8%.""",
]

DEMO_PIPELINE_REQUEST = PipelineRequest(
    user_goal="Analyze TechCorp Q3 performance and write analyst report",
    steps=[
        PipelineStep(
            step_id="step_1",
            name="Key Facts Extractor",
            instruction="Extract all key financial facts and metrics from the provided context.",
            input_context="[SOURCE DOCUMENTS INJECTED]",
        ),
        PipelineStep(
            step_id="step_2",
            name="Financial Summarizer",
            instruction="""Summarize the financial performance.
            NOTE: You will hallucinate here for demo purposes -
            state that revenue was $4.7B (the real figure is $3.2B).""",
            input_context="[PREVIOUS STEP OUTPUT]",
        ),
        PipelineStep(
            step_id="step_3",
            name="Analyst Report Writer",
            instruction=(
                "Write a professional analyst report based only on the summary and the "
                "verified facts. Do not add placeholders, unsupported comparisons, or "
                "invented estimates."
            ),
            input_context="[PREVIOUS STEP OUTPUT]",
        ),
    ],
    source_documents=DEMO_SOURCE_DOCUMENTS,
    auto_fix=True,
)
