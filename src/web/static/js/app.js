// Call Fraud Detection UI Client-side Logic
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("drop-zone");
  const audioInput = document.getElementById("audio-input");
  const fileInfo = document.getElementById("file-info");
  const fileName = document.getElementById("file-name");
  const fileSize = document.getElementById("file-size");
  const audioPlayer = document.getElementById("audio-player");
  const btnPredict = document.getElementById("btn-predict");
  const btnTranscribe = document.getElementById("btn-transcribe");
  const spinnerPredict = document.getElementById("spinner-predict");
  const spinnerTranscribe = document.getElementById("spinner-transcribe");

  const resultsCard = document.getElementById("results-card");
  const callIdBadge = document.getElementById("call-id-badge");
  const resOverall = document.getElementById("res-overall");
  const resProb = document.getElementById("res-prob");
  const resProbBar = document.getElementById("res-prob-bar");
  const resConfidence = document.getElementById("res-confidence");
  const resReview = document.getElementById("res-review");
  const riskList = document.getElementById("risk-list");
  const transcriptBody = document.getElementById("transcript-body");
  const transcriptLang = document.getElementById("transcript-lang");
  const btnCopyTranscript = document.getElementById("btn-copy-transcript");
  const segmentSection = document.getElementById("segment-section");
  const segmentsTableBody = document.getElementById("segments-table-body");
  const resDisclaimer = document.getElementById("res-disclaimer");

  const systemStatus = document.getElementById("system-status");
  const alertBanner = document.getElementById("alert-banner");
  const alertTitle = document.getElementById("alert-title");
  const alertMessage = document.getElementById("alert-message");
  const alertClose = document.getElementById("alert-close");

  let selectedFile = null;

  // 1. Health check poll
  async function checkSystemHealth() {
    try {
      const resp = await fetch("/api/v1/health");
      if (resp.ok) {
        const data = await resp.json();
        if (data.model_loaded) {
          systemStatus.className = "status-badge status-healthy";
          systemStatus.innerHTML = `<span class="status-dot"></span> Online (Model Active: ${data.model_version || "BiLSTM"})`;
        } else {
          systemStatus.className = "status-badge status-degraded";
          systemStatus.innerHTML = `<span class="status-dot"></span> System Degraded (Model Not Trained)`;
        }
      } else {
        throw new Error(`Health status ${resp.status}`);
      }
    } catch (err) {
      systemStatus.className = "status-badge status-degraded";
      systemStatus.innerHTML = `<span class="status-dot"></span> API Offline`;
    }
  }

  checkSystemHealth();
  setInterval(checkSystemHealth, 15000);

  // 2. Alert helpers
  function showError(title, message) {
    alertTitle.textContent = title;
    alertMessage.textContent = message;
    alertBanner.classList.remove("hidden");
  }

  function hideError() {
    alertBanner.classList.add("hidden");
  }

  alertClose.addEventListener("click", hideError);

  // 3. File selection
  function handleFile(file) {
    hideError();
    if (!file) return;

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

    const fileUrl = URL.createObjectURL(file);
    audioPlayer.src = fileUrl;
    fileInfo.classList.remove("hidden");

    btnPredict.disabled = false;
    btnTranscribe.disabled = false;
  }

  audioInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  // Drag and drop
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // 4. Prediction submit
  btnPredict.addEventListener("click", async () => {
    if (!selectedFile) return;
    hideError();

    const mode = document.querySelector('input[name="predict-mode"]:checked').value;
    const endpoint = mode === "segments" ? "/api/v1/predict/segments" : "/api/v1/predict";

    btnPredict.disabled = true;
    btnTranscribe.disabled = true;
    spinnerPredict.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const resp = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.detail || data.message || `Request failed with status ${resp.status}`);
      }

      displayResults(data);
    } catch (err) {
      showError("Prediction Failed", err.message);
    } finally {
      btnPredict.disabled = false;
      btnTranscribe.disabled = false;
      spinnerPredict.classList.add("hidden");
    }
  });

  // 5. Transcribe only submit
  btnTranscribe.addEventListener("click", async () => {
    if (!selectedFile) return;
    hideError();

    btnPredict.disabled = true;
    btnTranscribe.disabled = true;
    spinnerTranscribe.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const resp = await fetch("/api/v1/transcribe", {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.detail || `Transcription failed with status ${resp.status}`);
      }

      // Display transcription only in result box
      resultsCard.classList.remove("hidden");
      callIdBadge.textContent = `call_id: ${data.call_id}`;
      transcriptLang.textContent = data.language || "en";
      transcriptBody.textContent = data.transcript || "(No speech detected)";
      resOverall.textContent = "Transcription Only";
      resOverall.className = "kpi-value badge-status";
      resProb.textContent = "N/A";
      resProbBar.style.width = "0%";
      resConfidence.textContent = "N/A";
      resReview.textContent = "N/A";
      riskList.innerHTML = `<li class="empty-hint">Classification not performed in transcribe-only mode.</li>`;
      segmentSection.classList.add("hidden");
    } catch (err) {
      showError("Transcription Failed", err.message);
    } finally {
      btnPredict.disabled = false;
      btnTranscribe.disabled = false;
      spinnerTranscribe.classList.add("hidden");
    }
  });

  // 6. Display results
  function displayResults(data) {
    resultsCard.classList.remove("hidden");
    callIdBadge.textContent = `call_id: ${data.call_id}`;

    // Overall prediction status
    const pred = (data.overall_prediction || "uncertain").toLowerCase();
    resOverall.textContent = pred.toUpperCase();
    resOverall.className = `kpi-value badge-status status-${pred}`;

    // Probability
    const prob = Number(data.fraud_probability || 0);
    const pct = Math.round(prob * 100);
    resProb.textContent = `${pct}%`;
    resProbBar.style.width = `${pct}%`;
    if (prob >= 0.7) {
      resProbBar.style.backgroundColor = "var(--danger)";
    } else if (prob >= 0.4) {
      resProbBar.style.backgroundColor = "var(--warning)";
    } else {
      resProbBar.style.backgroundColor = "var(--success)";
    }

    // Confidence
    resConfidence.textContent = (data.confidence || "medium").toUpperCase();

    // Human review
    if (data.review_required) {
      resReview.textContent = "REQUIRED";
      resReview.style.color = "var(--warning)";
    } else {
      resReview.textContent = "Standard";
      resReview.style.color = "var(--text-main)";
    }

    // Transcript
    transcriptLang.textContent = data.language || "en";
    transcriptBody.textContent = data.transcript || "(No transcript text)";

    // Risk indicators
    riskList.innerHTML = "";
    if (data.risk_indicators && data.risk_indicators.length > 0) {
      data.risk_indicators.forEach((indicator) => {
        const li = document.createElement("li");
        li.textContent = `⚠️ ${indicator}`;
        riskList.appendChild(li);
      });
    } else {
      riskList.innerHTML = `<li class="empty-hint">No significant fraud linguistic markers detected.</li>`;
    }

    // Segments
    if (data.segments && data.segments.length > 0) {
      segmentSection.classList.remove("hidden");
      segmentsTableBody.innerHTML = "";
      data.segments.forEach((seg) => {
        const tr = document.createElement("tr");
        const segPct = Math.round(seg.fraud_probability * 100);
        tr.innerHTML = `
          <td><strong>${seg.start_time}s – ${seg.end_time}s</strong></td>
          <td>${seg.duration}s</td>
          <td><span class="badge ${seg.fraud_probability >= 0.7 ? "status-fraud" : seg.fraud_probability >= 0.4 ? "status-uncertain" : "status-legitimate"}">${segPct}%</span></td>
          <td><em>${seg.annotation_type}</em></td>
          <td>${seg.transcript_snippet || "-"}</td>
        `;
        segmentsTableBody.appendChild(tr);
      });
    } else {
      segmentSection.classList.add("hidden");
    }

    // Disclaimer
    if (data.disclaimer) {
      resDisclaimer.textContent = data.disclaimer;
    }

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // 7. Copy transcript
  btnCopyTranscript.addEventListener("click", () => {
    const text = transcriptBody.textContent;
    if (text) {
      navigator.clipboard.writeText(text);
      btnCopyTranscript.textContent = "✓ Copied!";
      setTimeout(() => {
        btnCopyTranscript.textContent = "📋 Copy";
      }, 2000);
    }
  });
});
