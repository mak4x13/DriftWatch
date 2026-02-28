const STORAGE_KEY = "driftwatch.savedPipelines";
const EMPTY_OUTPUT_MARKUP = "<p>Final pipeline output will appear here once the run finishes.</p>";

const demoDocuments = [
  "TechCorp Q3 2024 Earnings: Revenue reached $3.2B, up 12% YoY. CEO stated: expansion into Asian markets drove growth.",
  "Market Analysis: TechCorp faces headwinds from rising chip costs. Analyst consensus price target: $142. Current P/E ratio: 24x.",
  "Competitor Report: RivalCorp posted $2.8B revenue same quarter. TechCorp maintains 53% market share in enterprise segment.",
  "TechCorp Product Update: New AI chip released Q3, volume sales expected Q1 2026. R&D spend increased 18% this quarter.",
  "Economic Context: Fed rate unchanged. Tech sector up 4% this quarter. Inflation at 2.8%.",
];

const templateDefinitions = {
  financial_research: {
    label: "Financial Research Report",
    userGoal: "Analyze TechCorp Q3 performance and write analyst report",
    sourceDocuments: demoDocuments,
    steps: [
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
        instruction:
          "Write a professional analyst report based only on the summary and the verified facts. Do not add unsupported comparisons or invented estimates.",
      },
    ],
    driftwatchEnabled: true,
  },
  legal_summary: {
    label: "Legal Document Summary",
    userGoal: "Summarize the contract obligations, deadlines, and risk clauses for a legal reviewer",
    sourceDocuments: [
      "Master Services Agreement, Section 3: Vendor must deliver monthly compliance reports by the fifth business day of each month. Either party may terminate for uncured material breach after 30 days written notice.",
      "Data Processing Addendum, Section 7: Personal data must be deleted within 45 days of contract termination unless retention is required by law.",
      "Statement of Work, Milestones: Phase 1 acceptance testing closes on June 30, 2026. Final invoice is payable net 30 after written acceptance.",
    ],
    steps: [
      {
        name: "Clause Extractor",
        instruction:
          "Extract the contractual obligations, deadlines, termination rights, and payment terms from the provided legal materials.",
      },
      {
        name: "Risk Summarizer",
        instruction:
          "Summarize the highest-risk legal obligations and note where deadlines or termination clauses could affect the client.",
      },
      {
        name: "Reviewer Memo Writer",
        instruction:
          "Write a concise legal reviewer memo using only the grounded contract facts and clearly label any operational risk.",
      },
    ],
    driftwatchEnabled: true,
  },
  content_fact_checker: {
    label: "Content Fact-Checker",
    userGoal: "Verify a draft article against provided source notes and produce a corrected publish-ready version",
    sourceDocuments: [
      "Source Note 1: The city opened the Riverfront Innovation Center on March 12, 2026. Mayor Elena Park announced the center supports 42 local startups.",
      "Source Note 2: The center cost $18M to build and was funded through municipal bonds plus a state innovation grant.",
      "Source Note 3: The article should note that phase two of the project begins in September 2026 and is still pending approval.",
    ],
    steps: [
      {
        name: "Fact Extractor",
        instruction:
          "Extract all verifiable dates, names, figures, and pending milestones from the source notes.",
      },
      {
        name: "Draft Fact Checker",
        instruction:
          "Check a draft article against the extracted facts and identify any unsupported claims or missing qualifiers.",
      },
      {
        name: "Corrected Article Writer",
        instruction:
          "Write a publish-ready corrected article using only verified facts from the source notes and clearly label pending items.",
      },
    ],
    driftwatchEnabled: true,
  },
  custom_blank: {
    label: "Custom (blank)",
    userGoal: "",
    sourceDocuments: [""],
    steps: [
      {
        name: "",
        instruction: "",
      },
    ],
    driftwatchEnabled: true,
  },
};

const state = {
  eventSource: null,
  runId: null,
  events: [],
  result: null,
  running: false,
  currentSteps: [],
  stepStatus: {},
  protectionEnabled: true,
  activeTemplate: "financial_research",
};

const elements = {
  alertBanner: document.getElementById("alertBanner"),
  helpToggle: document.getElementById("helpToggle"),
  helpSidebar: document.getElementById("helpSidebar"),
  helpCloseButton: document.getElementById("helpCloseButton"),
  liveBadge: document.getElementById("liveBadge"),
  liveBadgeState: document.getElementById("liveBadgeState"),
  resetDemoButton: document.getElementById("resetDemoButton"),
  templateSelect: document.getElementById("templateSelect"),
  savePipelineButton: document.getElementById("savePipelineButton"),
  userGoal: document.getElementById("userGoal"),
  sourceDocuments: document.getElementById("sourceDocuments"),
  stepsContainer: document.getElementById("stepsContainer"),
  stepTemplate: document.getElementById("stepTemplate"),
  addStepButton: document.getElementById("addStepButton"),
  protectionToggle: document.getElementById("protectionToggle"),
  protectionState: document.getElementById("protectionState"),
  runDemoButton: document.getElementById("runDemoButton"),
  runCustomButton: document.getElementById("runCustomButton"),
  streamStatus: document.getElementById("streamStatus"),
  stepTimeline: document.getElementById("stepTimeline"),
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
  configureMarked();
  rebuildTemplateOptions();
  applyPipelineConfig(templateDefinitions.financial_research);
  bindEvents();
  resetRunState(null);
  updateProtectionState();
});

function configureMarked() {
  if (window.marked) {
    window.marked.setOptions({
      breaks: true,
      gfm: true,
    });
  }
}

function bindEvents() {
  elements.addStepButton.addEventListener("click", () => addStepCard());
  elements.templateSelect.addEventListener("change", handleTemplateSelection);
  elements.savePipelineButton.addEventListener("click", savePipeline);
  elements.protectionToggle.addEventListener("change", () => {
    state.protectionEnabled = elements.protectionToggle.checked;
    updateProtectionState();
  });
  elements.runDemoButton.addEventListener("click", runDemoPipeline);
  elements.runCustomButton.addEventListener("click", runCustomPipeline);
  elements.resetDemoButton.addEventListener("click", resetDemoState);
  elements.exportButton.addEventListener("click", exportAuditTrail);
  elements.helpToggle.addEventListener("click", toggleHelpSidebar);
  elements.helpCloseButton.addEventListener("click", closeHelpSidebar);
}

function handleTemplateSelection() {
  const value = elements.templateSelect.value;
  if (!value) {
    return;
  }

  if (value.startsWith("saved:")) {
    const savedPipeline = getSavedPipelines().find((entry) => `saved:${entry.id}` === value);
    if (savedPipeline) {
      applyPipelineConfig(savedPipeline);
      state.activeTemplate = value;
    }
    return;
  }

  const template = templateDefinitions[value];
  if (!template) {
    return;
  }

  applyPipelineConfig(template);
  state.activeTemplate = value;
}

function applyPipelineConfig(config) {
  elements.userGoal.value = config.userGoal || "";
  elements.sourceDocuments.value = (config.sourceDocuments || []).join("\n\n");
  elements.stepsContainer.innerHTML = "";
  (config.steps || []).forEach((step) => addStepCard(step));
  if (!config.steps || !config.steps.length) {
    addStepCard();
  }

  state.protectionEnabled =
    config.driftwatchEnabled !== undefined ? Boolean(config.driftwatchEnabled) : true;
  elements.protectionToggle.checked = state.protectionEnabled;
  updateProtectionState();
  updateStepIndices();
}

function addStepCard(step = { name: "", instruction: "" }) {
  const fragment = elements.stepTemplate.content.cloneNode(true);
  const card = fragment.querySelector(".step-card");
  const nameInput = fragment.querySelector(".step-name");
  const instructionInput = fragment.querySelector(".step-instruction");
  const removeButton = fragment.querySelector(".remove-step");

  nameInput.value = step.name || "";
  instructionInput.value = step.instruction || "";
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
  const demoTemplate = templateDefinitions.financial_research;
  applyPipelineConfig({
    ...demoTemplate,
    driftwatchEnabled: state.protectionEnabled,
  });

  const runId = generateRunId();
  const steps = buildStepsFromTemplate(demoTemplate.steps);
  const sourceDocuments = [...demoTemplate.sourceDocuments];
  const userGoal = demoTemplate.userGoal;

  await executePipeline({
    runId,
    steps,
    url:
      `/api/pipeline/demo?run_id=${encodeURIComponent(runId)}` +
      `&driftwatch_enabled=${encodeURIComponent(String(state.protectionEnabled))}`,
    fetchOptions: { method: "POST" },
  });
}

async function runCustomPipeline() {
  const steps = collectSteps();
  const sourceDocuments = parseSourceDocuments(elements.sourceDocuments.value);
  const userGoal = elements.userGoal.value.trim();

  if (!userGoal || !steps.length || !sourceDocuments.length) {
    showError(
      "Provide a user goal, at least one source document, and at least one complete step."
    );
    return;
  }

  const runId = generateRunId();
  const payload = {
    run_id: runId,
    user_goal: userGoal,
    steps,
    source_documents: sourceDocuments,
    auto_fix: true,
    driftwatch_enabled: state.protectionEnabled,
  };

  await executePipeline({
    runId,
    steps,
    url: "/api/pipeline/run",
    fetchOptions: {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  });
}

async function executePipeline({ runId, steps, url, fetchOptions }) {
  hideError();
  resetRunState(runId, steps);
  setRunning(true);
  connectEventStream(runId);

  try {
    const response = await fetch(url, fetchOptions);
    if (!response.ok) {
      throw new Error(await parseApiError(response));
    }

    const result = await response.json();
    state.result = result;
    renderPipelineResult(result);
    elements.streamStatus.textContent = "Pipeline run completed.";
  } catch (error) {
    showError(error.message || "Pipeline run failed.");
    renderSyntheticError(error);
    elements.streamStatus.textContent = "Pipeline run failed.";
  } finally {
    setRunning(false);
  }
}

function connectEventStream(runId) {
  closeEventStream();

  state.eventSource = new EventSource(`/api/pipeline/stream/${encodeURIComponent(runId)}`);
  state.eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    state.events.push(payload);
    renderEvent(payload);
    updateTimelineFromEvent(payload);
    if (payload.event_type === "PIPELINE_DONE" || payload.event_type === "ERROR") {
      closeEventStream();
    }
  };
  state.eventSource.onerror = closeEventStream;
}

function renderEvent(event) {
  const card = document.createElement("article");
  const issueClass = primaryIssueClass(event);
  card.className = `event-card ${classNameForEvent(event)} ${issueClass}`.trim();
  card.innerHTML = renderEventMarkup(event);

  elements.eventFeed.appendChild(card);
  activateCardGauges(card);
  elements.eventFeed.scrollTop = elements.eventFeed.scrollHeight;
}

function renderEventMarkup(event) {
  if (event.event_type === "STEP_START") {
    const stepName = event.data.step_name || event.step_id;
    elements.streamStatus.textContent = `Running ${stepName}...`;
    return renderStepEventCard({
      badge: renderStepBadge(event),
      title: `Running ${escapeHtml(stepName)}`,
      body: renderMarkdownBlock(event.data.instruction || "", { compact: true }),
      timestamp: event.timestamp,
    });
  }

  if (event.event_type === "STEP_COMPLETE") {
    return renderStepEventCard({
      badge: renderStepBadge(event),
      title: "Step Complete",
      body: renderMarkdownBlock(event.data.output_preview || "", { compact: true }),
      timestamp: event.timestamp,
    });
  }

  if (event.event_type === "AUDIT_RESULT") {
    const audit = event.data;
    const isPass = audit.verdict === "PASS";
    return renderStepEventCard({
      badge: renderStepBadge(event),
      title: `${isPass ? "PASS" : "FLAGGED"} - ${renderInlineMarkdown(audit.summary || "")}`,
      body: `
        ${renderInlineGauge(audit.drift_score || 0)}
        ${isPass ? "" : renderIssuesMarkup(audit.issues || [])}
      `,
      timestamp: event.timestamp,
      metaBadge: `Drift ${Number(audit.drift_score || 0).toFixed(2)}`,
    });
  }

  if (event.event_type === "AUTOFIX") {
    return renderStepEventCard({
      badge: renderStepBadge(event),
      title: "AUTO-FIX APPLIED",
      body: renderMarkdownBlock(event.data.summary || "", { compact: true }),
      timestamp: event.timestamp,
    });
  }

  if (event.event_type === "PIPELINE_DONE") {
    const data = event.data;
    return renderStepEventCard({
      badge: '<div class="step-pill pipeline-pill">Pipeline Summary</div>',
      title: data.pipeline_trustworthy ? "PIPELINE TRUSTWORTHY" : "REVIEW REQUIRED",
      body: `
        ${renderInlineGauge(data.overall_drift_score || 0)}
        <p>${escapeHtml(
          `${data.total_hallucinations} hallucination(s) caught, ${data.total_corrections} correction(s) applied.`
        )}</p>
      `,
      timestamp: event.timestamp,
      metaBadge: escapeHtml(data.overall_verdict || "REVIEW_REQUIRED"),
    });
  }

  return renderStepEventCard({
    badge: renderStepBadge(event),
    title: "Pipeline Error",
    body: `<p>${escapeHtml(event.data.message || "Unknown error")}</p>`,
    timestamp: event.timestamp,
  });
}

function renderStepEventCard({ badge, title, body, timestamp, metaBadge = "" }) {
  return `
    ${badge}
    <h3>${title}</h3>
    ${body}
    <div class="event-meta">
      <span>${formatTimestamp(timestamp)}</span>
      ${metaBadge ? `<span class="drift-badge">${metaBadge}</span>` : ""}
    </div>
  `;
}

function renderStepBadge(event) {
  if (event.step_id === "pipeline") {
    return '<div class="step-pill pipeline-pill">Pipeline</div>';
  }

  const step = findStepById(event.step_id);
  const stepName = event.data.step_name || step?.name || event.step_id;
  const stepNumber = extractStepNumber(event.step_id);
  return `<div class="step-pill">Step ${stepNumber} - ${escapeHtml(stepName)}</div>`;
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

function primaryIssueClass(event) {
  if (event.event_type !== "AUDIT_RESULT" || !event.data.issues || !event.data.issues.length) {
    return "";
  }

  const type = String(event.data.issues[0].type || "").toLowerCase().replaceAll("_", "-");
  return `issue-card-${type}`;
}

function renderIssuesMarkup(issues) {
  if (!issues.length) {
    return "";
  }

  const items = issues
    .map((issue) => {
      const typeClass = `issue-${String(issue.type || "").toLowerCase().replaceAll("_", "-")}`;
      return `
        <li class="issue-item ${typeClass}">
          <div class="issue-type-row">
            <strong>${escapeHtml(issue.type)}</strong>
            <span class="issue-severity">${escapeHtml(issue.severity || "LOW")}</span>
          </div>
          <div class="issue-text">${renderInlineMarkdown(issue.claim || "")}</div>
          <div class="issue-reason">${renderInlineMarkdown(issue.reason || "")}</div>
        </li>
      `;
    })
    .join("");
  return `<ul class="issue-list">${items}</ul>`;
}

function renderInlineGauge(score) {
  return `
    <div class="inline-gauge" aria-hidden="true">
      <div class="inline-gauge-fill" data-score="${Number(score || 0)}"></div>
    </div>
  `;
}

function activateCardGauges(card) {
  card.querySelectorAll(".inline-gauge-fill").forEach((fill) => {
    animateGauge(fill, Number(fill.dataset.score || 0));
  });
}

function updateTimelineFromEvent(event) {
  if (!state.currentSteps.length) {
    return;
  }

  if (event.event_type === "STEP_START") {
    state.stepStatus[event.step_id] = "active";
  } else if (event.event_type === "AUDIT_RESULT") {
    state.stepStatus[event.step_id] = event.data.verdict === "FLAG" ? "flag" : "pass";
  } else if (event.event_type === "ERROR" && event.step_id !== "pipeline") {
    state.stepStatus[event.step_id] = "flag";
  }

  renderTimeline();
}

function renderTimeline() {
  if (!state.currentSteps.length) {
    elements.stepTimeline.innerHTML = '<p class="timeline-empty">Pipeline steps will appear here.</p>';
    return;
  }

  elements.stepTimeline.innerHTML = state.currentSteps
    .map((step, index) => {
      const status = state.stepStatus[step.step_id] || "pending";
      return `
        <div class="timeline-item timeline-${status}">
          <div class="timeline-line"></div>
          <div class="timeline-circle">${timelineGlyph(status, index + 1)}</div>
          <div class="timeline-copy">
            <div class="timeline-title">Step ${index + 1}</div>
            <div class="timeline-name">${escapeHtml(step.name)}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

function timelineGlyph(status, index) {
  if (status === "pass") return "✓";
  if (status === "flag") return "X";
  return String(index);
}

function renderPipelineResult(result) {
  elements.statSteps.textContent = result.audit_results.length;
  elements.statHallucinations.textContent = result.total_hallucinations;
  elements.statCorrections.textContent = result.total_corrections;
  elements.driftScoreLabel.textContent = Number(result.overall_drift_score || 0).toFixed(2);
  animateGauge(elements.driftGaugeFill, Number(result.overall_drift_score || 0));

  elements.trustBadge.className = `trust-badge ${
    result.pipeline_trustworthy ? "trustworthy" : "review"
  }`;
  elements.trustBadge.textContent = result.pipeline_trustworthy
    ? "PIPELINE TRUSTWORTHY"
    : "REVIEW REQUIRED";

  const unresolvedClaims = result.audit_results
    .filter((entry) => entry.verdict === "FLAG" && !entry.auto_fixed)
    .flatMap((entry) => entry.issues || []);

  elements.finalOutput.innerHTML = renderMarkdownBlock(result.final_output || "", {
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

function resetRunState(runId, steps = state.currentSteps) {
  state.runId = runId;
  state.events = [];
  state.result = null;
  state.currentSteps = steps || [];
  state.stepStatus = Object.fromEntries((state.currentSteps || []).map((step) => [step.step_id, "pending"]));

  elements.eventFeed.innerHTML = "";
  elements.streamStatus.textContent = runId
    ? `Listening for events on run ${runId}...`
    : "Waiting for pipeline run.";
  elements.statSteps.textContent = "0";
  elements.statHallucinations.textContent = "0";
  elements.statCorrections.textContent = "0";
  elements.driftScoreLabel.textContent = "0.00";
  animateGauge(elements.driftGaugeFill, 0);
  elements.trustBadge.className = "trust-badge neutral";
  elements.trustBadge.textContent = "Awaiting run";
  elements.finalOutput.innerHTML = EMPTY_OUTPUT_MARKUP;
  elements.exportButton.disabled = true;
  renderTimeline();
}

function setRunning(isRunning) {
  state.running = isRunning;
  elements.runDemoButton.disabled = isRunning;
  elements.runCustomButton.disabled = isRunning;
  elements.addStepButton.disabled = isRunning;
  elements.savePipelineButton.disabled = isRunning;
  elements.templateSelect.disabled = isRunning;
  elements.resetDemoButton.disabled = isRunning;
  elements.protectionToggle.disabled = isRunning;
  elements.liveBadge.classList.toggle("running", isRunning);
  elements.liveBadge.classList.toggle("idle", !isRunning);
  elements.liveBadgeState.textContent = isRunning ? "Live" : "Idle";
}

function updateProtectionState() {
  elements.protectionState.textContent = state.protectionEnabled ? "ON" : "OFF";
  elements.protectionState.className = `protection-state ${
    state.protectionEnabled ? "protection-on" : "protection-off"
  }`;
}

function animateGauge(fillElement, score) {
  if (!fillElement) {
    return;
  }

  const clamped = Math.max(0, Math.min(1, Number(score || 0)));
  fillElement.style.transition = "none";
  fillElement.style.width = "0%";
  fillElement.style.background = severityColor(clamped);
  fillElement.dataset.severity = driftSeverity(clamped);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      fillElement.style.transition = "width 600ms ease, background-color 300ms ease";
      fillElement.style.width = `${clamped * 100}%`;
      fillElement.style.background = severityColor(clamped);
    });
  });
}

function driftSeverity(score) {
  if (score < 0.3) return "low";
  if (score < 0.6) return "medium";
  return "high";
}

function severityColor(score) {
  const severity = driftSeverity(score);
  if (severity === "low") return "#45d483";
  if (severity === "medium") return "#ffb454";
  return "#ff6b6b";
}

function exportAuditTrail() {
  if (!state.result) {
    return;
  }

  const timestamp = state.events.find((event) => event.event_type === "PIPELINE_DONE")?.timestamp
    || new Date().toISOString();
  const payload = {
    run_id: state.result.run_id,
    user_goal: state.result.user_goal,
    timestamp,
    overall_verdict:
      state.result.overall_verdict
      || (state.result.pipeline_trustworthy ? "TRUSTWORTHY" : "REVIEW_REQUIRED"),
    overall_drift_score: state.result.overall_drift_score,
    total_hallucinations: state.result.total_hallucinations,
    total_corrections: state.result.total_corrections,
    steps: state.result.audit_results.map((entry) => ({
      step_name: entry.step_name || findStepById(entry.step_id)?.name || entry.step_id,
      step_id: entry.step_id,
      original_output: entry.original_output,
      final_output: entry.final_output,
      audit_verdict: entry.verdict,
      drift_score: entry.drift_score,
      issues_found: entry.issues || [],
      auto_fixed: Boolean(entry.auto_fixed),
    })),
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `driftwatch-${state.runId}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function resetDemoState() {
  hideError();
  try {
    const response = await fetch("/api/demo/reset", { method: "POST" });
    if (!response.ok) {
      throw new Error(await parseApiError(response));
    }
    resetRunState(null, state.currentSteps);
    elements.streamStatus.textContent = "Demo state reset successfully.";
  } catch (error) {
    showError(error.message || "Unable to reset the demo state.");
  }
}

function closeEventStream() {
  if (!state.eventSource) {
    return;
  }

  state.eventSource.close();
  state.eventSource = null;
}

function renderMarkdownBlock(text, options = {}) {
  const source = String(text || "").trim();
  if (!source) {
    return "<p>No output returned.</p>";
  }

  const markedSource = applyClaimMarkers(source, options.highlightClaims || []);
  if (window.marked) {
    const html = window.marked.parse(markedSource);
    const compactClass = options.compact ? " compact" : "";
    return `<div class="rich-text${compactClass}">${html}</div>`;
  }

  return `<div class="rich-text"><p>${escapeHtml(source)}</p></div>`;
}

function renderInlineMarkdown(text) {
  const source = String(text || "").trim();
  if (!source) {
    return "";
  }

  if (window.marked && typeof window.marked.parseInline === "function") {
    return window.marked.parseInline(source);
  }
  return escapeHtml(source);
}

function applyClaimMarkers(text, issues) {
  let markedText = String(text);
  issues.forEach((issue) => {
    if (!issue || !issue.claim) {
      return;
    }
    markedText = markedText.split(issue.claim).join(
      `<span class="claim-highlight">${issue.claim}</span>`
    );
  });
  return markedText;
}

function showError(message) {
  elements.alertBanner.hidden = false;
  elements.alertBanner.textContent = message;
}

function hideError() {
  elements.alertBanner.hidden = true;
  elements.alertBanner.textContent = "";
}

function toggleHelpSidebar() {
  const open = elements.helpSidebar.classList.toggle("open");
  elements.helpSidebar.setAttribute("aria-hidden", String(!open));
  elements.helpToggle.setAttribute("aria-expanded", String(open));
}

function closeHelpSidebar() {
  elements.helpSidebar.classList.remove("open");
  elements.helpSidebar.setAttribute("aria-hidden", "true");
  elements.helpToggle.setAttribute("aria-expanded", "false");
}

function getSavedPipelines() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (error) {
    return [];
  }
}

function savePipeline() {
  const steps = collectSteps();
  const sourceDocuments = parseSourceDocuments(elements.sourceDocuments.value);
  const userGoal = elements.userGoal.value.trim();
  const name = window.prompt("Name this pipeline");

  if (!name) {
    return;
  }

  const savedPipelines = getSavedPipelines();
  savedPipelines.push({
    id: generateRunId(),
    label: name.trim(),
    userGoal,
    sourceDocuments,
    steps: steps.map((step) => ({
      name: step.name,
      instruction: step.instruction,
    })),
    driftwatchEnabled: state.protectionEnabled,
  });

  localStorage.setItem(STORAGE_KEY, JSON.stringify(savedPipelines));
  rebuildTemplateOptions(`saved:${savedPipelines[savedPipelines.length - 1].id}`);
}

function rebuildTemplateOptions(selectedValue = state.activeTemplate) {
  const savedPipelines = getSavedPipelines();
  elements.templateSelect.innerHTML = "";

  const templateGroup = document.createElement("optgroup");
  templateGroup.label = "Templates";
  Object.entries(templateDefinitions).forEach(([value, template]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = template.label;
    templateGroup.appendChild(option);
  });
  elements.templateSelect.appendChild(templateGroup);

  if (savedPipelines.length) {
    const savedGroup = document.createElement("optgroup");
    savedGroup.label = "Saved Pipelines";
    savedPipelines.forEach((entry) => {
      const option = document.createElement("option");
      option.value = `saved:${entry.id}`;
      option.textContent = entry.label;
      savedGroup.appendChild(option);
    });
    elements.templateSelect.appendChild(savedGroup);
  }

  elements.templateSelect.value = selectedValue;
}

function buildStepsFromTemplate(steps) {
  return steps.map((step, index) => ({
    step_id: `step_${index + 1}`,
    name: step.name,
    instruction: step.instruction,
    input_context: index === 0 ? "[SOURCE DOCUMENTS INJECTED]" : "[PREVIOUS STEP OUTPUT]",
  }));
}

function parseApiError(response) {
  return response
    .json()
    .then((payload) => payload.detail || payload.message || `Request failed with ${response.status}`)
    .catch(() => `Request failed with ${response.status}`);
}

function findStepById(stepId) {
  return state.currentSteps.find((step) => step.step_id === stepId);
}

function extractStepNumber(stepId) {
  const match = String(stepId || "").match(/(\d+)/);
  return match ? match[1] : "?";
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
