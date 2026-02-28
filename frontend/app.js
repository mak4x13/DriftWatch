const demoDocuments = [
  "TechCorp Q3 2024 Earnings: Revenue reached $3.2B, up 12% YoY. CEO stated: expansion into Asian markets drove growth.",
  "Market Analysis: TechCorp faces headwinds from rising chip costs. Analyst consensus price target: $142. Current P/E ratio: 24x.",
  "Competitor Report: RivalCorp posted $2.8B revenue same quarter. TechCorp maintains 53% market share in enterprise segment.",
  "TechCorp Product Update: New AI chip released Q3, volume sales expected Q1 2026. R&D spend increased 18% this quarter.",
  "Economic Context: Fed rate unchanged. Tech sector up 4% this quarter. Inflation at 2.8%.",
];

const demoSteps = [
  {
    name: "Key Facts Extractor",
    instruction: "Extract all key financial facts and metrics from the provided context.",
  },
  {
    name: "Financial Summarizer",
    instruction:
      "Summarize the financial performance. NOTE: You will hallucinate here for demo purposes - state that revenue was $4.7B (the real figure is $3.2B).",
  },
  {
    name: "Analyst Report Writer",
    instruction: "Write a professional analyst report based on the summary.",
  },
];

const state = {
  eventSource: null,
  runId: null,
  events: [],
  result: null,
  running: false,
};

const elements = {
  userGoal: document.getElementById("userGoal"),
  sourceDocuments: document.getElementById("sourceDocuments"),
  stepsContainer: document.getElementById("stepsContainer"),
  stepTemplate: document.getElementById("stepTemplate"),
  addStepButton: document.getElementById("addStepButton"),
  runDemoButton: document.getElementById("runDemoButton"),
  runCustomButton: document.getElementById("runCustomButton"),
  streamStatus: document.getElementById("streamStatus"),
  eventFeed: document.getElementById("eventFeed"),
  statSteps: document.getElementById("statSteps"),
  statHallucinations: document.getElementById("statHallucinations"),
  statCorrections: document.getElementById("statCorrections"),
  driftScoreLabel: document.getElementById("driftScoreLabel"),
  driftGaugeFill: document.getElementById("driftGaugeFill"),
  trustBadge: document.getElementById("trustBadge"),
  finalOutput: document.getElementById("finalOutput"),
  exportButton: document.getElementById("exportButton"),
};

document.addEventListener("DOMContentLoaded", () => {
  seedDemoForm();
  bindEvents();
  updateStepIndices();
  resetRunState(null);
});

function bindEvents() {
  elements.addStepButton.addEventListener("click", () => addStepCard());
  elements.runDemoButton.addEventListener("click", runDemoPipeline);
  elements.runCustomButton.addEventListener("click", runCustomPipeline);
  elements.exportButton.addEventListener("click", exportAuditTrail);
}

function seedDemoForm() {
  elements.userGoal.value = "Analyze TechCorp Q3 performance and write analyst report";
  elements.sourceDocuments.value = demoDocuments.join("\n\n");
  elements.stepsContainer.innerHTML = "";
  demoSteps.forEach((step) => addStepCard(step));
}

function addStepCard(step = { name: "", instruction: "" }) {
  const fragment = elements.stepTemplate.content.cloneNode(true);
  const card = fragment.querySelector(".step-card");
  const nameInput = fragment.querySelector(".step-name");
  const instructionInput = fragment.querySelector(".step-instruction");
  const removeButton = fragment.querySelector(".remove-step");

  nameInput.value = step.name;
  instructionInput.value = step.instruction;
  removeButton.addEventListener("click", () => {
    card.remove();
    updateStepIndices();
  });

  elements.stepsContainer.appendChild(fragment);
  updateStepIndices();
}

function updateStepIndices() {
  const cards = [...elements.stepsContainer.querySelectorAll(".step-card")];
  cards.forEach((card, index) => {
    const badge = card.querySelector(".step-index");
    badge.textContent = index + 1;
  });
}

function collectSteps() {
  return [...elements.stepsContainer.querySelectorAll(".step-card")]
    .map((card, index) => {
      const name = card.querySelector(".step-name").value.trim();
      const instruction = card.querySelector(".step-instruction").value.trim();
      if (!name || !instruction) {
        return null;
      }
      return {
        step_id: `step_${index + 1}`,
        name,
        instruction,
        input_context: index === 0 ? "[SOURCE DOCUMENTS INJECTED]" : "[PREVIOUS STEP OUTPUT]",
      };
    })
    .filter(Boolean);
}

function parseSourceDocuments(rawValue) {
  return rawValue
    .split(/\n\s*\n/g)
    .map((entry) => entry.trim())
    .filter(Boolean);
}

async function runDemoPipeline() {
  seedDemoForm();
  const runId = generateRunId();
  await executePipeline({
    runId,
    url: `/api/pipeline/demo?run_id=${encodeURIComponent(runId)}`,
    fetchOptions: { method: "POST" },
  });
}

async function runCustomPipeline() {
  const steps = collectSteps();
  const sourceDocuments = parseSourceDocuments(elements.sourceDocuments.value);
  if (!elements.userGoal.value.trim() || !steps.length || !sourceDocuments.length) {
    elements.streamStatus.textContent =
      "Provide a user goal, at least one source document, and at least one complete step.";
    return;
  }

  const runId = generateRunId();
  const payload = {
    run_id: runId,
    user_goal: elements.userGoal.value.trim(),
    steps,
    source_documents: sourceDocuments,
    auto_fix: true,
  };

  await executePipeline({
    runId,
    url: "/api/pipeline/run",
    fetchOptions: {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  });
}

async function executePipeline({ runId, url, fetchOptions }) {
  resetRunState(runId);
  setRunning(true);
  connectEventStream(runId);

  try {
    const response = await fetch(url, fetchOptions);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Request failed with ${response.status}`);
    }

    const result = await response.json();
    state.result = result;
    renderPipelineResult(result);
    elements.streamStatus.textContent = "Pipeline run completed.";
  } catch (error) {
    renderSyntheticError(error);
    elements.streamStatus.textContent = "Pipeline run failed.";
  } finally {
    setRunning(false);
  }
}

function connectEventStream(runId) {
  if (state.eventSource) {
    state.eventSource.close();
  }

  state.eventSource = new EventSource(`/api/pipeline/stream/${encodeURIComponent(runId)}`);
  state.eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    state.events.push(payload);
    renderEvent(payload);
    if (payload.event_type === "PIPELINE_DONE" || payload.event_type === "ERROR") {
      state.eventSource.close();
    }
  };
  state.eventSource.onerror = () => {
    if (state.eventSource) {
      state.eventSource.close();
    }
  };
}

function renderEvent(event) {
  const card = document.createElement("article");
  card.className = `event-card ${classNameForEvent(event)}`;

  if (event.event_type === "STEP_START") {
    card.innerHTML = `
      <h3>&#9654; Running: ${escapeHtml(event.data.step_name || event.step_id)}</h3>
      <p>${escapeHtml(event.data.instruction || "")}</p>
      <div class="event-meta"><span>${formatTimestamp(event.timestamp)}</span></div>
    `;
    elements.streamStatus.textContent = `Running ${event.data.step_name || event.step_id}...`;
  }

  if (event.event_type === "STEP_COMPLETE") {
    card.innerHTML = `
      <h3>Step Complete</h3>
      ${renderRichText(event.data.output_preview || "", { className: "compact" })}
      <div class="event-meta"><span>${formatTimestamp(event.timestamp)}</span></div>
    `;
  }

  if (event.event_type === "AUDIT_RESULT") {
    const audit = event.data;
    const isPass = audit.verdict === "PASS";
    card.innerHTML = `
      <h3>${isPass ? "&#10003; PASS" : "&#9888; FLAGGED"} - ${renderInlineMarkup(
        audit.summary || ""
      )}</h3>
      ${isPass ? "" : renderIssuesMarkup(audit.issues || [])}
      <div class="event-meta">
        <span>${formatTimestamp(event.timestamp)}</span>
        <span class="drift-badge">Drift ${Number(audit.drift_score || 0).toFixed(2)}</span>
      </div>
    `;
  }

  if (event.event_type === "AUTOFIX") {
    card.innerHTML = `
      <h3>&#128295; AUTO-FIX APPLIED</h3>
      ${renderRichText(event.data.summary || "", { className: "compact" })}
      <div class="event-meta"><span>${formatTimestamp(event.timestamp)}</span></div>
    `;
  }

  if (event.event_type === "PIPELINE_DONE") {
    const data = event.data;
    card.innerHTML = `
      <h3>${data.pipeline_trustworthy ? "&#10003; PIPELINE TRUSTWORTHY" : "&#9888; REVIEW REQUIRED"}</h3>
      <p>${escapeHtml(
        `${data.total_hallucinations} hallucination(s) caught, ${data.total_corrections} correction(s) applied.`
      )}</p>
      <div class="event-meta">
        <span>${formatTimestamp(event.timestamp)}</span>
        <span class="drift-badge">Max Drift ${Number(data.overall_drift_score || 0).toFixed(2)}</span>
      </div>
    `;
  }

  if (event.event_type === "ERROR") {
    card.innerHTML = `
      <h3>Pipeline Error</h3>
      <p>${escapeHtml(event.data.message || "Unknown error")}</p>
      <div class="event-meta"><span>${formatTimestamp(event.timestamp)}</span></div>
    `;
  }

  elements.eventFeed.appendChild(card);
  elements.eventFeed.scrollTop = elements.eventFeed.scrollHeight;
}

function classNameForEvent(event) {
  if (event.event_type === "STEP_START") return "event-start";
  if (event.event_type === "STEP_COMPLETE") return "event-step";
  if (event.event_type === "AUDIT_RESULT" && event.data.verdict === "PASS") return "event-pass";
  if (event.event_type === "AUDIT_RESULT" && event.data.verdict === "FLAG") return "event-flag";
  if (event.event_type === "AUTOFIX") return "event-autofix";
  if (event.event_type === "PIPELINE_DONE") return "event-done";
  return "event-error";
}

function renderIssuesMarkup(issues) {
  if (!issues.length) {
    return "";
  }

  const items = issues
    .map(
      (issue) => `
        <li>
          <strong>${escapeHtml(issue.type)}</strong>: ${renderInlineMarkup(issue.claim || "")}<br />
          ${renderInlineMarkup(issue.reason || "")}
        </li>
      `
    )
    .join("");
  return `<ul>${items}</ul>`;
}

function renderPipelineResult(result) {
  elements.statSteps.textContent = result.audit_results.length;
  elements.statHallucinations.textContent = result.total_hallucinations;
  elements.statCorrections.textContent = result.total_corrections;
  elements.driftScoreLabel.textContent = Number(result.overall_drift_score || 0).toFixed(2);
  elements.driftGaugeFill.style.width = `${Math.min(
    100,
    Number(result.overall_drift_score || 0) * 100
  )}%`;

  elements.trustBadge.className = `trust-badge ${
    result.pipeline_trustworthy ? "trustworthy" : "review"
  }`;
  elements.trustBadge.innerHTML = result.pipeline_trustworthy
    ? "&#10003; PIPELINE TRUSTWORTHY"
    : "&#9888; REVIEW REQUIRED";

  const unresolvedClaims = result.audit_results
    .filter((entry) => entry.verdict === "FLAG" && !entry.auto_fixed)
    .flatMap((entry) => entry.issues || []);

  elements.finalOutput.innerHTML = renderRichText(result.final_output || "", {
    highlightClaims: unresolvedClaims,
  });
  elements.exportButton.disabled = false;
}

function renderSyntheticError(error) {
  renderEvent({
    event_type: "ERROR",
    step_id: "pipeline",
    data: { message: error.message || String(error) },
    timestamp: new Date().toISOString(),
  });
}

function resetRunState(runId) {
  state.runId = runId;
  state.events = [];
  state.result = null;
  elements.eventFeed.innerHTML = "";
  elements.streamStatus.textContent = runId
    ? `Listening for events on run ${runId}...`
    : "Waiting for pipeline run.";
  elements.statSteps.textContent = "0";
  elements.statHallucinations.textContent = "0";
  elements.statCorrections.textContent = "0";
  elements.driftScoreLabel.textContent = "0.00";
  elements.driftGaugeFill.style.width = "0%";
  elements.trustBadge.className = "trust-badge neutral";
  elements.trustBadge.textContent = "Awaiting run";
  elements.finalOutput.innerHTML = "<p>Final pipeline output will appear here once the run finishes.</p>";
  elements.exportButton.disabled = true;
}

function setRunning(isRunning) {
  state.running = isRunning;
  elements.runDemoButton.disabled = isRunning;
  elements.runCustomButton.disabled = isRunning;
  elements.addStepButton.disabled = isRunning;
}

function exportAuditTrail() {
  if (!state.result) {
    return;
  }

  const blob = new Blob(
    [
      JSON.stringify(
        {
          run_id: state.runId,
          result: state.result,
          events: state.events,
        },
        null,
        2
      ),
    ],
    { type: "application/json" }
  );
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `driftwatch-${state.runId}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function renderRichText(text, options = {}) {
  const source = String(text || "").trim();
  if (!source) {
    return "<p>No output returned.</p>";
  }

  const marked = applyClaimMarkers(source, options.highlightClaims || []);
  const className = options.className ? `rich-text ${options.className}` : "rich-text";
  return `<div class="${className}">${markdownToHtml(marked)}</div>`;
}

function markdownToHtml(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      html.push(`<h${level}>${renderInlineMarkup(headingMatch[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (isHorizontalRule(trimmed)) {
      html.push("<hr />");
      index += 1;
      continue;
    }

    if (isTableStart(lines, index)) {
      const [tableHtml, nextIndex] = renderTable(lines, index);
      html.push(tableHtml);
      index = nextIndex;
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const [listHtml, nextIndex] = renderList(lines, index, "ul");
      html.push(listHtml);
      index = nextIndex;
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const [listHtml, nextIndex] = renderList(lines, index, "ol");
      html.push(listHtml);
      index = nextIndex;
      continue;
    }

    const paragraphLines = [];
    while (index < lines.length) {
      const candidate = lines[index];
      const candidateTrimmed = candidate.trim();
      if (!candidateTrimmed) {
        break;
      }
      if (
        candidateTrimmed.match(/^(#{1,4})\s+/) ||
        isHorizontalRule(candidateTrimmed) ||
        /^\s*[-*+]\s+/.test(candidate) ||
        /^\s*\d+\.\s+/.test(candidate) ||
        isTableStart(lines, index)
      ) {
        break;
      }
      paragraphLines.push(candidateTrimmed);
      index += 1;
    }

    html.push(`<p>${paragraphLines.map((entry) => renderInlineMarkup(entry)).join("<br />")}</p>`);
  }

  return html.join("");
}

function renderList(lines, startIndex, tagName) {
  const items = [];
  let index = startIndex;
  const pattern = tagName === "ol" ? /^\s*\d+\.\s+(.*)$/ : /^\s*[-*+]\s+(.*)$/;

  while (index < lines.length) {
    const match = lines[index].match(pattern);
    if (!match) {
      break;
    }
    items.push(`<li>${renderInlineMarkup(match[1].trim())}</li>`);
    index += 1;
  }

  return [`<${tagName}>${items.join("")}</${tagName}>`, index];
}

function isTableStart(lines, index) {
  return index + 1 < lines.length && lines[index].includes("|") && isTableSeparator(lines[index + 1]);
}

function isTableSeparator(line) {
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);
}

function isHorizontalRule(line) {
  return /^\s*(?:---+|\*\*\*+|___+)\s*$/.test(line);
}

function renderTable(lines, startIndex) {
  const headers = splitTableRow(lines[startIndex]);
  const rows = [];
  let index = startIndex + 2;

  while (index < lines.length) {
    const candidate = lines[index];
    if (!candidate.trim() || !candidate.includes("|")) {
      break;
    }
    const cells = splitTableRow(candidate);
    if (cells.length < 2) {
      break;
    }
    rows.push(cells);
    index += 1;
  }

  const headerHtml = headers.map((cell) => `<th>${renderInlineMarkup(cell)}</th>`).join("");
  const bodyHtml = rows
    .map(
      (row) =>
        `<tr>${row.map((cell) => `<td>${renderInlineMarkup(cell)}</td>`).join("")}</tr>`
    )
    .join("");

  return [`<table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`, index];
}

function splitTableRow(line) {
  let trimmed = line.trim();
  if (trimmed.startsWith("|")) {
    trimmed = trimmed.slice(1);
  }
  if (trimmed.endsWith("|")) {
    trimmed = trimmed.slice(0, -1);
  }
  return trimmed.split("|").map((cell) => cell.trim());
}

function applyClaimMarkers(text, issues) {
  let marked = text;
  issues.forEach((issue, index) => {
    if (!issue || !issue.claim) {
      return;
    }
    const startToken = `@@HL${index}_START@@`;
    const endToken = `@@HL${index}_END@@`;
    marked = marked.split(issue.claim).join(`${startToken}${issue.claim}${endToken}`);
  });
  return marked;
}

function renderInlineMarkup(text) {
  let html = escapeHtml(text);
  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noreferrer">$1</a>'
  );
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  html = html.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
  html = html.replace(
    /@@HL(\d+)_START@@([\s\S]*?)@@HL\1_END@@/g,
    '<span class="claim-highlight">$2</span>'
  );
  return html;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatTimestamp(value) {
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function generateRunId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `run-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
