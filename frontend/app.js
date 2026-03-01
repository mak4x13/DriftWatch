const EMPTY_OUTPUT_MARKUP = "<p>Final pipeline output will appear here once the run finishes.</p>";

const state = {
  eventSource: null,
  runId: null,
  events: [],
  result: null,
  running: false,
  currentSteps: [],
  stepStatus: {},
  protectionEnabled: true,
  uploadedFiles: [],
  uploadDragDepth: 0,
};

const elements = {
  alertBanner: document.getElementById("alertBanner"),
  helpToggle: document.getElementById("helpToggle"),
  helpSidebar: document.getElementById("helpSidebar"),
  helpCloseButton: document.getElementById("helpCloseButton"),
  liveBadge: document.getElementById("liveBadge"),
  liveBadgeState: document.getElementById("liveBadgeState"),
  resetDemoButton: document.getElementById("resetDemoButton"),
  userGoal: document.getElementById("userGoal"),
  sourceDocuments: document.getElementById("sourceDocuments"),
  uploadZone: document.getElementById("uploadZone"),
  fileInput: document.getElementById("fileInput"),
  uploadedFileChips: document.getElementById("uploadedFileChips"),
  protectionToggle: document.getElementById("protectionToggle"),
  protectionState: document.getElementById("protectionState"),
  runDemoButton: document.getElementById("runDemoButton"),
  runCustomButton: document.getElementById("runCustomButton"),
  streamStatus: document.getElementById("streamStatus"),
  streamScrollArea: document.getElementById("streamScrollArea"),
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
  configurePdfJs();
  bindEvents();
  resetRunState(null);
  updateProtectionState();
  renderUploadedFileChips();
});

function configureMarked() {
  if (window.marked) {
    window.marked.setOptions({
      breaks: true,
      gfm: true,
    });
  }
}

function configurePdfJs() {
  if (window.pdfjsLib && window.pdfjsLib.GlobalWorkerOptions) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  }
}

function bindEvents() {
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
  bindUploadEvents();
}

function bindUploadEvents() {
  elements.uploadZone.addEventListener("click", (event) => {
    event.preventDefault();
    elements.fileInput.click();
  });

  elements.uploadZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.fileInput.click();
    }
  });

  elements.fileInput.addEventListener("change", async (event) => {
    await handleUploadedFiles(event.target.files);
    elements.fileInput.value = "";
  });

  elements.fileInput.addEventListener("click", (event) => {
    event.stopPropagation();
  });

  elements.uploadZone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    event.stopPropagation();
    state.uploadDragDepth += 1;
    elements.uploadZone.classList.add("upload-zone-active");
  });

  elements.uploadZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadZone.classList.add("upload-zone-active");
  });

  ["dragleave", "dragend"].forEach((eventName) => {
    elements.uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      state.uploadDragDepth = Math.max(0, state.uploadDragDepth - 1);
      if (state.uploadDragDepth === 0) {
        elements.uploadZone.classList.remove("upload-zone-active");
      }
    });
  });

  elements.uploadZone.addEventListener("drop", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    state.uploadDragDepth = 0;
    elements.uploadZone.classList.remove("upload-zone-active");
    await handleUploadedFiles(event.dataTransfer?.files);
  });

  elements.uploadedFileChips.addEventListener("click", (event) => {
    const removeButton = event.target.closest("[data-remove-upload]");
    if (!removeButton) {
      return;
    }
    removeUploadedFile(removeButton.dataset.removeUpload);
  });
}

function parseSourceDocuments(rawValue) {
  return rawValue
    .split(/\n\s*\n/g)
    .map((entry) => entry.trim())
    .filter(Boolean);
}

async function handleUploadedFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) {
    return;
  }

  hideError();
  elements.uploadZone.classList.add("upload-zone-busy");

  const results = await Promise.allSettled(files.map((file) => readUploadedFile(file)));
  const failures = [];

  results.forEach((result, index) => {
    if (result.status === "fulfilled") {
      appendUploadedDocument(files[index], result.value);
      return;
    }
    failures.push(files[index]?.name || `file ${index + 1}`);
  });

  elements.uploadZone.classList.remove("upload-zone-busy");

  if (failures.length) {
    showError(`Unable to read: ${failures.join(", ")}. Upload .txt, .md, or .pdf files only.`);
  }
}

async function readUploadedFile(file) {
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  if (extension === "txt" || extension === "md") {
    return readTextFile(file);
  }
  if (extension === "pdf") {
    return extractPdfText(file);
  }
  throw new Error(`Unsupported file type: ${file.name}`);
}

function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error(`Unable to read ${file.name}`));
    reader.readAsText(file);
  });
}

async function extractPdfText(file) {
  if (!window.pdfjsLib) {
    throw new Error("PDF parser failed to load.");
  }

  const buffer = await file.arrayBuffer();
  const pdf = await window.pdfjsLib.getDocument({ data: buffer }).promise;
  const pageTexts = [];

  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const textContent = await page.getTextContent();
    const pageText = textContent.items
      .map((item) => String(item.str || "").trim())
      .filter(Boolean)
      .join(" ");
    pageTexts.push(pageText);
  }

  return pageTexts.join("\n\n");
}

function appendUploadedDocument(file, content) {
  const normalizedContent = String(content || "").replace(/\r\n?/g, "\n").trim();
  if (!normalizedContent) {
    return;
  }

  const fileLabel = nextUploadedFileLabel(file.name);
  const block = buildUploadedFileBlock(fileLabel, normalizedContent);
  const existingValue = elements.sourceDocuments.value.trimEnd();
  elements.sourceDocuments.value = existingValue ? `${existingValue}\n\n${block}` : block;

  state.uploadedFiles.push({
    id: cryptoSafeId(),
    name: file.name,
    label: fileLabel,
  });
  renderUploadedFileChips();
}

function nextUploadedFileLabel(fileName) {
  const baseMatches = state.uploadedFiles.filter((entry) => entry.name === fileName).length;
  return baseMatches ? `${fileName} (${baseMatches + 1})` : fileName;
}

function buildUploadedFileBlock(fileLabel, content) {
  return [
    `----- Uploaded File: ${fileLabel} -----`,
    content,
    `----- End Uploaded File: ${fileLabel} -----`,
  ].join("\n");
}

function renderUploadedFileChips() {
  if (!state.uploadedFiles.length) {
    elements.uploadedFileChips.innerHTML = "";
    elements.uploadedFileChips.hidden = true;
    return;
  }

  elements.uploadedFileChips.hidden = false;
  elements.uploadedFileChips.innerHTML = state.uploadedFiles
    .map(
      (file) => `
        <div class="file-chip">
          <span class="file-chip-name">${escapeHtml(file.label)}</span>
          <button
            type="button"
            class="file-chip-remove"
            data-remove-upload="${escapeHtml(file.id)}"
            aria-label="Remove ${escapeHtml(file.label)}"
          >
            ×
          </button>
        </div>
      `
    )
    .join("");
}

function removeUploadedFile(fileId) {
  const file = state.uploadedFiles.find((entry) => entry.id === fileId);
  if (!file) {
    return;
  }

  elements.sourceDocuments.value = removeUploadedBlock(elements.sourceDocuments.value, file.label);
  state.uploadedFiles = state.uploadedFiles.filter((entry) => entry.id !== fileId);
  renderUploadedFileChips();
}

function removeUploadedBlock(sourceText, fileLabel) {
  const escapedLabel = escapeRegExp(fileLabel);
  const blockPattern = new RegExp(
    `(?:\\n{0,2})----- Uploaded File: ${escapedLabel} -----\\n[\\s\\S]*?\\n----- End Uploaded File: ${escapedLabel} -----(?:\\n{0,2})`,
    "g"
  );
  return String(sourceText || "")
    .replace(blockPattern, "\n\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function cryptoSafeId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `upload-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function connectEventStream(runId) {
  closeEventStream();

  state.eventSource = new EventSource(`/api/pipeline/stream/${encodeURIComponent(runId)}`);
  state.eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    registerStepFromEvent(payload);
    state.events.push(payload);
    renderEvent(payload);
    updateTimelineFromEvent(payload);
    if (payload.event_type === "PIPELINE_DONE" || payload.event_type === "ERROR") {
      closeEventStream();
    }
  };
  state.eventSource.onerror = closeEventStream;
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

function legacyTimelineGlyph(status, index) {
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

  elements.trustBadge.className = `trust-badge ${trustBadgeClass(result.overall_verdict)}`;
  elements.trustBadge.textContent = trustBadgeLabel(result.overall_verdict);

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

  const markedSource = applyClaimMarkers(
    normalizeOrderedListNumbering(source),
    options.highlightClaims || []
  );
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

function normalizeOrderedListNumbering(text) {
  const lines = String(text || "").split("\n");
  let orderedIndex = 0;

  return lines
    .map((line) => {
      const match = line.match(/^(\s*)(\d+)\.\s+(.*)$/);
      if (!match) {
        orderedIndex = 0;
        return line;
      }

      orderedIndex += 1;
      return `${match[1]}${orderedIndex}. ${match[3]}`;
    })
    .join("\n");
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

function buildPipelineSteps(steps) {
  return (steps || [])
    .map((step) => {
      const stepName = String(step.step_name || step.name || "").trim();
      const instruction = String(step.instruction || "").trim();
      if (!stepName || !instruction) {
        return null;
      }
      return {
        step_name: stepName,
        instruction,
      };
    })
    .filter(Boolean)
    .map((step, index) => ({
    step_id: `step_${index + 1}`,
    name: step.step_name,
    instruction: step.instruction,
    input_context: index === 0 ? "[SOURCE DOCUMENTS INJECTED]" : "[PREVIOUS STEP OUTPUT]",
  }));
}

function clearActiveStepCards(stepId) {
  [...elements.eventFeed.querySelectorAll(".event-card.event-loading")].forEach((card) => {
    if (card.dataset.stepId === stepId) {
      card.classList.remove("event-loading");
    }
  });
}

function renderCardLoader() {
  return `
    <span class="card-loader" aria-hidden="true">
      <span></span><span></span><span></span>
    </span>
  `;
}

function renderTimelineDots() {
  return `
    <span class="timeline-dots" aria-hidden="true">
      <span></span><span></span><span></span>
    </span>
  `;
}

async function runDemoPipeline() {
  const runId = generateRunId();
  await executePipeline({
    runId,
    steps: [],
    url:
      `/api/pipeline/demo?run_id=${encodeURIComponent(runId)}` +
      `&driftwatch_enabled=${encodeURIComponent(String(state.protectionEnabled))}`,
    fetchOptions: { method: "POST" },
  });
}

async function runCustomPipeline() {
  const userGoal = elements.userGoal.value.trim();
  const sourceDocuments = parseSourceDocuments(elements.sourceDocuments.value);

  if (!userGoal || !sourceDocuments.length) {
    showError("Provide a user goal and at least one source document before running.");
    return;
  }

  const runId = generateRunId();

  hideError();
  setRunning(true);
  elements.streamStatus.textContent = "Generating custom pipeline steps...";

  try {
    const generatedSteps = await generatePipelineSteps({
      runId,
      userGoal,
      sourceDocuments,
    });

    const steps = buildPipelineSteps(generatedSteps);
    await executePipeline({
      runId,
      steps,
      url: "/api/pipeline/run",
      fetchOptions: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: runId,
          user_goal: userGoal,
          steps,
          source_documents: sourceDocuments,
          auto_fix: true,
          driftwatch_enabled: state.protectionEnabled,
        }),
      },
      keepRunning: true,
    });
  } catch (error) {
    elements.streamStatus.textContent = "Pipeline step generation failed.";
    showError(error.message || "Unable to generate pipeline steps.");
    renderSyntheticError(error);
  } finally {
    setRunning(false);
  }
}

async function generatePipelineSteps({ runId, userGoal, sourceDocuments }) {
  const response = await fetch("/api/pipeline/generate-steps", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run_id: runId,
      user_goal: userGoal,
      source_documents: sourceDocuments,
    }),
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  const payload = await response.json();
  const steps = buildPipelineSteps(payload.steps || []).map((step) => ({
    step_name: step.name,
    instruction: step.instruction,
  }));
  if (!steps.length) {
    throw new Error("No pipeline steps were generated.");
  }
  return steps;
}

async function executePipeline({ runId, steps, url, fetchOptions, keepRunning = false }) {
  hideError();
  resetRunState(runId, steps);
  if (!keepRunning) {
    setRunning(true);
  }
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

function renderEvent(event) {
  if (event.event_type === "STEP_TOKEN") {
    updateStreamingCard(event);
    return;
  }

  if (event.step_id !== "pipeline" && event.event_type !== "STEP_START") {
    clearActiveStepCards(event.step_id);
  }

  const card = document.createElement("article");
  const issueClass = primaryIssueClass(event);
  card.className = `event-card ${classNameForEvent(event)} ${issueClass}`.trim();
  card.dataset.stepId = event.step_id;
  card.dataset.eventType = event.event_type;
  if (event.event_type === "STEP_START") {
    card.classList.add("event-loading");
  }
  card.innerHTML = renderEventMarkup(event);

  elements.eventFeed.appendChild(card);
  activateCardGauges(card);
  elements.streamScrollArea.scrollTop = elements.streamScrollArea.scrollHeight;
}

function renderEventMarkup(event) {
  if (event.event_type === "STEP_START") {
    const stepName = event.data.step_name || event.step_id;
    elements.streamStatus.textContent = `Running ${stepName}...`;
    return renderStepEventCard({
      badge: renderStepBadge(event),
      title: `Running ${escapeHtml(stepName)}`,
      body: `
        ${renderMarkdownBlock(event.data.instruction || "", { compact: true })}
        <div class="stream-preview" data-stream-preview="${escapeHtml(event.step_id)}">
          <p>Generating output...</p>
        </div>
      `,
      timestamp: event.timestamp,
      loading: true,
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
      title: trustBadgeLabel(data.overall_verdict),
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

function renderStepEventCard({ badge, title, body, timestamp, metaBadge = "", loading = false }) {
  return `
    ${badge}
    <h3 class="event-title">
      <span>${title}</span>
      ${loading ? renderCardLoader() : ""}
    </h3>
    ${body}
    <div class="event-meta">
      <span>${formatTimestamp(timestamp)}</span>
      ${metaBadge ? `<span class="drift-badge">${metaBadge}</span>` : ""}
    </div>
  `;
}

function updateStreamingCard(event) {
  const card = findLatestLoadingCard(event.step_id);
  if (!card) {
    return;
  }

  const preview = card.querySelector("[data-stream-preview]");
  if (preview) {
    preview.innerHTML = renderMarkdownBlock(event.data.content || "", { compact: true });
  }

  const stepName = event.data.step_name || findStepById(event.step_id)?.name || event.step_id;
  elements.streamStatus.textContent = `Streaming ${stepName}...`;
}

function findLatestLoadingCard(stepId) {
  const cards = [...elements.eventFeed.querySelectorAll(".event-card.event-loading")];
  return cards.reverse().find((card) => card.dataset.stepId === stepId) || null;
}

function registerStepFromEvent(event) {
  if (!event || event.step_id === "pipeline") {
    return;
  }

  const stepName = String(event.data?.step_name || "").trim();
  if (!stepName) {
    return;
  }

  if (findStepById(event.step_id)) {
    return;
  }

  state.currentSteps.push({
    step_id: event.step_id,
    name: stepName,
  });
  state.currentSteps.sort((left, right) => stepOrder(left.step_id) - stepOrder(right.step_id));
  state.stepStatus[event.step_id] = state.stepStatus[event.step_id] || "pending";
  renderTimeline();
}

function stepOrder(stepId) {
  const match = String(stepId || "").match(/(\d+)/);
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function updateTimelineFromEvent(event) {
  if (!state.currentSteps.length) {
    return;
  }

  if (event.event_type === "STEP_START") {
    state.stepStatus[event.step_id] = "active";
  } else if (event.event_type === "AUTOFIX") {
    state.stepStatus[event.step_id] = "fixed";
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
            <div class="timeline-name-row">
              <div class="timeline-name">${escapeHtml(step.name)}</div>
              ${status === "active" ? renderTimelineDots() : ""}
            </div>
          </div>
        </div>
      `;
    })
    .join("");
}

function timelineGlyph(status, index) {
  if (status === "active") return '<span class="timeline-spinner" aria-hidden="true"></span>';
  if (status === "pass") return '<span class="timeline-glyph">&#10003;</span>';
  if (status === "fixed") return '<span class="timeline-glyph timeline-wrench">&#128295;</span>';
  if (status === "flag") return '<span class="timeline-glyph">&#10005;</span>';
  return `<span class="timeline-glyph">${index}</span>`;
}

function trustBadgeLabel(verdict) {
  if (verdict === "TRUSTWORTHY") return "PIPELINE TRUSTWORTHY";
  if (verdict === "PARTIALLY_VERIFIED") return "PARTIALLY VERIFIED";
  return "REVIEW REQUIRED";
}

function trustBadgeClass(verdict) {
  if (verdict === "TRUSTWORTHY") return "trustworthy";
  if (verdict === "PARTIALLY_VERIFIED") return "partial";
  return "review";
}

function resetRunState(runId, steps = []) {
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
  elements.resetDemoButton.disabled = isRunning;
  elements.protectionToggle.disabled = isRunning;
  elements.liveBadge.classList.toggle("running", isRunning);
  elements.liveBadge.classList.toggle("idle", !isRunning);
  elements.liveBadgeState.textContent = isRunning ? "Live" : "Idle";
}

async function resetDemoState() {
  hideError();
  try {
    const response = await fetch("/api/demo/reset", { method: "POST" });
    if (!response.ok) {
      throw new Error(await parseApiError(response));
    }
    resetRunState(null);
    elements.streamStatus.textContent = "Demo state reset successfully.";
  } catch (error) {
    showError(error.message || "Unable to reset the demo state.");
  }
}
