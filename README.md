[![Built with Mistral](https://img.shields.io/badge/Built%20With-Mistral-ff7000)](https://mistral.ai/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Local-7B61FF)](https://www.trychroma.com/)
[![Hackathon Submission 2025](https://img.shields.io/badge/Hackathon%20Submission-2026-E94560)](#)

# DriftWatch - AI That Catches AI Lying

## The Problem
Hallucination is already a known weakness in large language models, but the more dangerous failure mode appears when AI systems are chained together. In a multi-step pipeline, an unsupported fact introduced early can silently pass into the next step, get rephrased as if it were true, and eventually show up in a final report with the confidence of a verified conclusion.

That compounding failure makes enterprise AI hard to trust. A research agent can invent a number, a summarizer can normalize it, and a writer can present it as analysis. By the time a human reviewer sees the output, the original error is buried. DriftWatch is built to catch that drift before it becomes a polished lie.

## What DriftWatch Does
DriftWatch is an open-source semantic audit layer for multi-step AI pipelines. It runs between agent steps and checks whether the latest output still serves the original goal, whether its concrete claims are grounded in source documents, and whether it contradicts earlier pipeline outputs.

Every step goes through an audit loop. First the step runs with Mistral. Then DriftWatch retrieves relevant source passages from ChromaDB, sends the output through an auditor prompt, and merges that with deterministic claim checks so obvious hallucinations do not slip through. If a step is flagged, DriftWatch can trigger an auto-fix rewrite before forwarding the cleaned output to the next stage.

That turns the system from a black-box chain into an inspectable process. Instead of trusting the final answer blindly, the user sees a live trail of which step ran, what it produced, what was flagged, what was corrected, and whether the full pipeline is trustworthy enough to use.

## Demo
[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red)]([URL])

## Architecture
```text
  +---------------------------------------------------------+
  |                  DriftWatch System                      |
  |                                                         |
  |  [Browser UI]  <------------------------------------+   |
  |       |                                            |    |
  |       | HTTP POST /api/pipeline/run                |    |
  |       v                                        SSE |    |
  |  [FastAPI Backend]                           stream|    |
  |       |                                            |    |
  |       +--- Step 1: AgentRunner ---> Mistral API    |    |
  |       |         (runs pipeline step)               |    |
  |       |              |                             |    |
  |       |              v                             |    | 
  |       +--- Step 2: Auditor ------> Mistral API     |    |
  |       |         (checks output vs sources)         |    |
  |       |              |                             |    |
  |       |         [PASS / FLAG]                      |    |
  |       |              |                             |    |
  |       |         FLAG? -> AutoFix -> Mistral API    |    |
  |       |              |                             |    |
  |       v              v                             |    |
  |  [ChromaDB]    [AuditLog] ------------------------>+    |
  |  (vector store  (JSON trail for UI)                     |
  |   for RAG)                                              |
  +---------------------------------------------------------+
```

## Hackathon Track
- Track: Track 2 - Anything Goes
- Challenge: Best Use of Mistral API
- Models: mistral-large-latest, mistral-embed

## Quick Start
```bash
git clone https://github.com/mak4x13/DriftWatch
cd DriftWatch
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Add your MISTRAL_API_KEY to .env
uvicorn backend.main:app --reload
# Open http://localhost:8000
```

## How It Works
1. Intent Drift Check: DriftWatch compares each step output with the original user goal to make sure the pipeline is still solving the right problem.
2. Hallucination Detection: DriftWatch retrieves relevant source passages from ChromaDB and verifies specific claims against them before the next step can reuse the output.
3. Contradiction Guard: DriftWatch compares later steps against earlier steps so an invented claim does not harden into the final answer.

## Why This Matters
Enterprises do not just need more capable models. They need ways to inspect, constrain, and trust agentic systems when those systems are asked to perform research, reporting, financial analysis, operations work, or other high-stakes tasks. DriftWatch targets the point where trust usually fails: the hidden handoff between one AI step and the next.

By making semantic drift visible and correctable, DriftWatch helps move AI pipelines from impressive demos toward usable infrastructure. It shows that model performance alone is not enough; real adoption requires auditability, grounding, and failure containment at every step.

## Built With
- Mistral AI API (mistral-large-latest, mistral-embed)
- FastAPI + Python 3.11
- ChromaDB (local vector store)
- Vanilla HTML, CSS, and JavaScript
