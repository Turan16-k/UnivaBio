// LittleHeart user interface.

const SITE_NAMES = { AV: "Aortic", PV: "Pulmonary", TV: "Tricuspid", MV: "Mitral", Phc: "Other site" };
const LABEL_TEXT = ["No murmur", "Murmur", "Unclear"];
const LABEL_CLASS = ["ok", "bad", "warn"];

let recordings = [];   // { id, site, name, url, result }
let currentCase = null;
let nextId = 1;

const $ = (sel) => document.querySelector(sel);
const pct = (x) => Math.round(x * 100) + "%";

// ------------------------------------------------------------ tabs

function showTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + name));
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
    showTab(tab.dataset.tab);
    history.replaceState(null, "", "#" + tab.dataset.tab);
  }));
  const fromUrl = location.hash.slice(1);
  if (document.getElementById("tab-" + fromUrl)) showTab(fromUrl);

  document.querySelectorAll(".source").forEach(btn => btn.addEventListener("click", () => {
    document.querySelectorAll(".source").forEach(b => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".source-panel").forEach(p =>
      p.classList.toggle("active", p.id === "source-" + btn.dataset.source));
  }));
}

// ------------------------------------------------------------ adding recordings

async function addRecording(arrayBuffer, site, name) {
  const id = nextId++;
  const url = URL.createObjectURL(new Blob([arrayBuffer]));
  const rec = { id, site, name, url, result: null };
  recordings.push(rec);
  renderAll();
  try {
    const samples = await LHFeatures.decodeToRate(arrayBuffer, LHModel.config?.features.sample_rate || 4000);
    rec.result = await LHModel.analyzeRecording(samples);
  } catch (err) {
    console.error(err);
    rec.error = "Could not read this audio file.";
  }
  renderAll();
}

function renderDemoCases() {
  const cases = METRICS.demo_cases;
  $("#demo-cases").innerHTML = cases.map((c, i) => `
    <button class="case" data-case="${i}">
      <span class="case-title">Case ${i + 1}</span>
      <span class="case-meta">${c.age}, ${c.sex.toLowerCase()} · ${c.recordings.length} recordings</span>
    </button>`).join("");
  document.querySelectorAll("[data-case]").forEach(btn => btn.addEventListener("click", () => loadCase(+btn.dataset.case)));
}

async function loadCase(index) {
  document.querySelectorAll("[data-case]").forEach(b => b.classList.toggle("active", +b.dataset.case === index));
  clearRecordings();
  currentCase = METRICS.demo_cases[index];
  const caseRef = currentCase;
  if (window.innerWidth < 980) $("#verdict-card").scrollIntoView({ behavior: "smooth", block: "start" });
  await LHModel.load();
  for (const r of caseRef.recordings) {
    if (currentCase !== caseRef) return;
    const buf = await (await fetch("samples/" + r.file)).arrayBuffer();
    await addRecording(buf, r.location, r.file);
  }
}

function clearRecordings() {
  recordings.forEach(r => URL.revokeObjectURL(r.url));
  recordings = [];
  currentCase = null;
  renderAll();
}

function siteFromName(name) {
  const m = name.match(/(?:^|[_\-\s])(AV|PV|TV|MV|Phc)(?:[_\-\s.]|$)/i);
  if (!m) return null;
  const s = m[1].toUpperCase();
  return s === "PHC" ? "Phc" : s;
}

function setupUpload() {
  const input = $("#file-input");
  const drop = $("#file-drop");
  const handle = async (files) => {
    if (currentCase) clearRecordings();
    const order = ["AV", "PV", "TV", "MV"];
    for (const file of files) {
      const used = new Set(recordings.map(r => r.site));
      const site = siteFromName(file.name) || order.find(s => !used.has(s)) || "Phc";
      await addRecording(await file.arrayBuffer(), site, file.name);
    }
  };
  input.addEventListener("change", () => { handle([...input.files]); input.value = ""; });
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", e => {
    e.preventDefault();
    drop.classList.remove("over");
    handle([...e.dataTransfer.files]);
  });
}

function setupRecorder() {
  const btn = $("#record-btn");
  const status = $("#record-status");
  const siteSelect = $("#record-site");
  siteSelect.addEventListener("change", () => highlightSite(siteSelect.value));

  btn.addEventListener("click", async () => {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch {
      status.textContent = "Microphone permission was denied.";
      return;
    }
    if (currentCase) clearRecordings();
    const chunks = [];
    const recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      btn.disabled = false;
      status.textContent = "Analysing...";
      const site = siteSelect.value;
      recordings.filter(r => r.site === site).forEach(r => URL.revokeObjectURL(r.url));
      recordings = recordings.filter(r => r.site !== site);
      await addRecording(await new Blob(chunks).arrayBuffer(), site, `Recording at ${site}`);
      status.textContent = "Done. Choose the next site and record again.";
    };
    recorder.start();
    btn.disabled = true;
    let left = 15;
    status.textContent = `Recording... ${left} s`;
    const timer = setInterval(() => {
      left -= 1;
      status.textContent = `Recording... ${left} s`;
      if (left <= 0) {
        clearInterval(timer);
        recorder.stop();
      }
    }, 1000);
  });
}

// ------------------------------------------------------------ rendering

function renderAll() {
  renderChest();
  renderVerdict();
  renderSiteResults();
}

function siteLabel(site) {
  const done = recordings.filter(r => r.site === site && r.result);
  if (!done.length) return recordings.some(r => r.site === site) ? "busy" : null;
  const labels = done.map(r => LHModel.recordingLabel(r.result));
  if (labels.includes(LHModel.PRESENT)) return LABEL_CLASS[1];
  if (labels.includes(LHModel.UNKNOWN)) return LABEL_CLASS[2];
  return LABEL_CLASS[0];
}

function renderChest() {
  document.querySelectorAll(".site").forEach(g => {
    g.setAttribute("class", "site " + (siteLabel(g.dataset.site) || ""));
  });
}

function highlightSite(site) {
  document.querySelectorAll(".site").forEach(g => g.classList.toggle("focus", g.dataset.site === site));
}

function renderVerdict() {
  const box = $("#verdict");
  const truth = $("#truth");
  const done = recordings.filter(r => r.result);
  const pending = recordings.length - done.length;
  truth.hidden = true;
  $("#verdict-actions").hidden = !done.length || pending > 0;

  if (!recordings.length) {
    box.className = "verdict empty";
    box.innerHTML = verdictHtml("♡", "No recordings yet", "Pick a real case or add recordings to start.");
    return;
  }
  if (!done.length) {
    box.className = "verdict empty";
    box.innerHTML = verdictHtml("…", "Analysing", "Turning sound into a spectrogram and running the AI on this device.");
    return;
  }

  const d = LHModel.decide(done.map(r => r.result));
  const trusted = done.filter(r => r.result.beat.ok);
  const strongest = trusted.length ? trusted.reduce((a, b) => (b.result.probs[1] > a.result.probs[1] ? b : a)) : done[0];
  const unclear = done.filter((r, i) => d.labels[i] === LHModel.UNKNOWN).map(r => r.site);
  const noBeat = done.filter(r => !r.result.beat.ok).map(r => r.site);
  const more = pending ? ` Still analysing ${pending} recording${pending > 1 ? "s" : ""}.` : "";

  if (d.label === LHModel.PRESENT) {
    box.className = "verdict bad";
    box.innerHTML = verdictHtml("!", "Murmur suspected: refer for echocardiogram",
      `Strongest at the ${SITE_NAMES[strongest.site].toLowerCase()} area (${pct(d.present)} murmur likelihood). Arrange an echocardiogram to check the heart's structure.${more}`);
  } else if (d.label === LHModel.UNKNOWN && noBeat.length === done.length) {
    box.className = "verdict warn";
    box.innerHTML = verdictHtml("?", "No heartbeat found: check and record again",
      `LittleHeart could not find a regular heartbeat in ${noBeat.length > 1 ? "these recordings" : "this recording"}, so it will not guess. Press the stethoscope firmly on bare skin at the marked spot and keep the room quiet.${more}`);
  } else if (d.label === LHModel.UNKNOWN) {
    box.className = "verdict warn";
    const why = noBeat.length ? ` No heartbeat rhythm was found at ${[...new Set(noBeat)].join(", ")}.` : "";
    box.innerHTML = verdictHtml("?", "Recording unclear: record again",
      `The AI could not judge ${[...new Set(unclear)].join(", ")} reliably.${why} Re-record in a quieter moment; if it stays unclear, refer to a clinician.${more}`);
  } else {
    box.className = "verdict ok";
    box.innerHTML = verdictHtml("✓", "No murmur detected",
      `Highest murmur likelihood ${pct(d.present)} (referral threshold ${pct(d.thresholds.present)}). This does not rule out heart disease; follow usual care.${more}`);
  }

  if (currentCase && !pending) {
    const truthLabel = ["Absent", "Present", "Unknown"].indexOf(currentCase.truth);
    const agree = truthLabel === d.label;
    const details = currentCase.truth === "Present"
      ? `murmur heard at ${currentCase.murmur_locations.join(", ")}${currentCase.timing !== "nan" ? ` · ${currentCase.timing.toLowerCase()}` : ""}${currentCase.grade !== "nan" ? `, grade ${currentCase.grade}` : ""}`
      : currentCase.truth === "Absent" ? "no murmur" : "the expert could not decide (unclear recording)";
    truth.hidden = false;
    truth.className = "truth " + (agree ? "agree" : "disagree");
    truth.innerHTML = `<b>Cardiologist's finding:</b> ${details}. <span>${agree ? "LittleHeart agreed ✓" : "LittleHeart disagreed ✗"}</span>`;
  }
}

function verdictHtml(icon, title, text) {
  return `<div class="verdict-icon" aria-hidden="true">${icon}</div>
    <div><div class="verdict-title">${title}</div><div class="verdict-text">${text}</div></div>`;
}

function renderSiteResults() {
  const container = $("#site-results");
  const order = { AV: 0, PV: 1, TV: 2, MV: 3, Phc: 4 };
  const sorted = [...recordings].sort((a, b) => order[a.site] - order[b.site] || a.id - b.id);
  const existing = new Map([...container.children].map(el => [+el.dataset.id, el]));

  sorted.forEach(rec => {
    let el = existing.get(rec.id);
    const state = rec.error ? "error" : rec.result ? "done" : "busy";
    if (!el || el.dataset.state !== state) {
      const fresh = buildSiteCard(rec, state);
      if (el) el.replaceWith(fresh); else container.appendChild(fresh);
      el = fresh;
    }
    container.appendChild(el);
    existing.delete(rec.id);
  });
  existing.forEach(el => el.remove());
}

function buildSiteCard(rec, state) {
  const el = document.createElement("div");
  el.className = "card site-card";
  el.dataset.id = rec.id;
  el.dataset.state = state;
  const title = `<div class="site-head"><h3>${SITE_NAMES[rec.site] || rec.site} <span class="site-code">${rec.site}</span></h3>`;

  if (state === "busy") {
    el.innerHTML = `${title}<span class="badge">analysing…</span></div><div class="skeleton"></div>`;
    return el;
  }
  if (state === "error") {
    el.innerHTML = `${title}<span class="badge warn">error</span></div><p class="small">${rec.error}</p>`;
    return el;
  }

  const r = rec.result;
  const t = LHModel.config.thresholds;
  const label = LHModel.recordingLabel(r);
  const badgeText = r.beat.ok ? LABEL_TEXT[label] : "No heartbeat found";
  el.innerHTML = `${title}<span class="badge ${LABEL_CLASS[label]}">${badgeText}</span></div>
    <div class="spec-wrap">
      <canvas class="spec" aria-label="Spectrogram"></canvas>
      <div class="playhead"></div>
      <span class="axis-label top">1000 Hz</span><span class="axis-label bottom">25 Hz</span>
    </div>
    <canvas class="timeline" aria-label="Murmur likelihood over time"></canvas>
    <div class="site-foot">
      <button class="play" aria-label="Play recording">▶</button>
      <span class="small muted">${r.duration.toFixed(1)} s · ${r.beat.ok ? "heartbeat ✓" : "no heartbeat rhythm"} · murmur <b>${pct(r.probs[1])}</b> · unclear ${pct(r.probs[2])}</span>
    </div>
    <audio preload="none" src="${rec.url}"></audio>`;

  drawSpectrogram(el.querySelector(".spec"), r.spec);
  requestAnimationFrame(() => drawTimeline(el.querySelector(".timeline"), r, t.present));
  setupPlayback(el, r.duration);
  return el;
}

// ------------------------------------------------------------ drawing

const COLOR_STOPS = [[13, 8, 35], [62, 24, 104], [158, 47, 127], [234, 105, 84], [252, 216, 141]];

function colorAt(v) {
  const x = Math.min(0.9999, Math.max(0, v)) * (COLOR_STOPS.length - 1);
  const i = Math.floor(x), f = x - i;
  return COLOR_STOPS[i].map((c, k) => Math.round(c + (COLOR_STOPS[i + 1][k] - c) * f));
}

function drawSpectrogram(canvas, spec) {
  const nMels = spec.data.length / spec.nFrames;
  canvas.width = spec.nFrames;
  canvas.height = nMels;
  const sorted = Float32Array.from(spec.data).sort();
  const lo = sorted[Math.floor(sorted.length * 0.3)];
  const hi = sorted[Math.floor(sorted.length * 0.995)];
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(spec.nFrames, nMels);
  for (let m = 0; m < nMels; m++) {
    for (let f = 0; f < spec.nFrames; f++) {
      const [r, g, b] = colorAt((spec.data[m * spec.nFrames + f] - lo) / (hi - lo));
      const p = ((nMels - 1 - m) * spec.nFrames + f) * 4;
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawTimeline(canvas, result, threshold) {
  const css = getComputedStyle(document.documentElement);
  const w = canvas.clientWidth || 600, h = 56;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const x = (sec) => (sec / result.duration) * w;
  const y = (p) => h - 4 - p * (h - 14);

  ctx.strokeStyle = css.getPropertyValue("--border");
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, y(threshold)); ctx.lineTo(w, y(threshold)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = css.getPropertyValue("--muted");
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("murmur likelihood per 4 s window · dashed = referral threshold", 4, 10);

  const span = Math.min(4, result.duration);
  result.windows.forEach(win => {
    const p = win.probs[1];
    ctx.fillStyle = p >= threshold ? css.getPropertyValue("--bad") : css.getPropertyValue("--ok");
    ctx.globalAlpha = 0.25 + 0.6 * p;
    const left = x(win.startSec + span / 2 - 1), right = x(Math.min(result.duration, win.startSec + span / 2 + 1));
    ctx.fillRect(left, y(p), Math.max(2, right - left - 1), h - 4 - y(p));
  });
  ctx.globalAlpha = 1;
}

function setupPlayback(el, duration) {
  const audio = el.querySelector("audio");
  const btn = el.querySelector(".play");
  const head = el.querySelector(".playhead");
  let raf = null;
  const tick = () => {
    head.style.left = (100 * audio.currentTime / duration) + "%";
    if (!audio.paused) raf = requestAnimationFrame(tick);
  };
  btn.addEventListener("click", () => {
    document.querySelectorAll("audio").forEach(a => { if (a !== audio) a.pause(); });
    if (audio.paused) audio.play(); else audio.pause();
  });
  audio.addEventListener("play", () => { btn.textContent = "❚❚"; head.style.display = "block"; tick(); });
  audio.addEventListener("pause", () => { btn.textContent = "▶"; cancelAnimationFrame(raf); });
  audio.addEventListener("ended", () => { btn.textContent = "▶"; head.style.display = "none"; });
}

// ------------------------------------------------------------ validation tab

function renderValidation() {
  const m = METRICS, c = m.cnn, ds = m.dataset;
  const pm = (k) => `${pct(c[k].mean)} <small>± ${Math.round(c[k].std * 100)}</small>`;
  $("#val-intro").textContent =
    `LittleHeart was tested with ${m.folds}-fold cross-validation on ${ds.cv_children} children (${ds.recordings.toLocaleString("en-US")} recordings in the full dataset). Each child was judged by a model that had never heard them.`;

  const cards = [
    [c.weighted_accuracy.mean.toFixed(2) + ` <small>± ${c.weighted_accuracy.std.toFixed(2)}</small>`, "Challenge score",
      "PhysioNet 2022 weighted accuracy: murmurs count 5×, unclear 3×, normal 1×."],
    [pm("sensitivity"), "Murmurs caught", "Children with a murmur that LittleHeart referred."],
    [pm("specificity"), "Normal hearts cleared", "Children without a murmur that were not referred."],
    [c.auc.mean.toFixed(2) + ` <small>± ${c.auc.std.toFixed(2)}</small>`, "AUC", "How well murmur likelihood ranks children with and without a murmur (1.0 = perfect)."],
  ];
  $("#val-cards").innerHTML = cards.map(([big, title, text]) =>
    `<div class="card metric"><div class="metric-big">${big}</div><h3>${title}</h3><p class="muted small">${text}</p></div>`).join("");

  const bars = [
    ["Simple baseline (ours)", m.baseline.weighted_accuracy.mean, "base"],
    ["LittleHeart CNN (ours)", c.weighted_accuracy.mean, "ours"],
    ["Best team, PhysioNet 2022*", m.challenge_reference.top_hidden_test, "ref"],
  ];
  $("#compare").innerHTML = bars.map(([name, v, cls]) => `
    <div class="cmp-row"><span class="cmp-name">${name}</span>
      <span class="cmp-track"><span class="cmp-bar ${cls}" style="width:${v * 100}%"></span></span>
      <span class="cmp-val">${v.toFixed(2)}</span></div>`).join("");
  $("#compare-note").textContent =
    "*The challenge's best score (team HearHeart, among 40 ranked murmur entries) was measured on a hidden test set we cannot access, so the comparison is indicative only.";

  const names = ["No murmur", "Murmur", "Unclear"];
  const cm = m.confusion;
  let html = `<div></div>${names.map(n => `<div class="cm-head">${n}</div>`).join("")}`;
  cm.forEach((row, t) => {
    html += `<div class="cm-side">${names[t]}</div>`;
    row.forEach((v, p) => { html += `<div class="cm-cell ${t === p ? "good" : "bad"}">${v}</div>`; });
  });
  $("#confusion").innerHTML = html;
  $("#miss-rate").textContent = Math.max(2, Math.round(1 / Math.max(0.01, 1 - c.sensitivity.mean)));
}

function renderSafety() {
  const b = LHModel.beatCheck;
  const rows = [[`Real heart recordings (${b.real_recordings.toLocaleString("en-US")})`, `${(b.real_pass_rate * 100).toFixed(1)}% accepted`, true]];
  Object.entries(b.negatives).forEach(([name, v]) =>
    rows.push([name[0].toUpperCase() + name.slice(1) + " (synthetic)", `${v.rejected}/${v.trials} rejected`, v.rejected === v.trials]));
  $("#safety").innerHTML = rows.map(([name, result, good]) =>
    `<div class="safety-row"><span>${name}</span><b class="${good ? "ok-text" : "warn-text"}">${result}</b></div>`).join("");
}

// ------------------------------------------------------------ start

setupTabs();
setupUpload();
setupRecorder();
renderDemoCases();
renderValidation();
renderAll();
$("#print-btn").addEventListener("click", () => {
  $("#print-meta").textContent = `LittleHeart screening note · ${new Date().toLocaleString("en-GB")} · ` +
    `${recordings.length} recording(s) · Research prototype, not a diagnosis. Referral decisions remain with a clinician.`;
  window.print();
});
$("#clear-btn").addEventListener("click", () => {
  document.querySelectorAll("[data-case]").forEach(b => b.classList.remove("active"));
  clearRecordings();
});
document.querySelectorAll(".site").forEach(g => g.addEventListener("click", () => {
  $("#record-site").value = g.dataset.site;
  highlightSite(g.dataset.site);
}));
const caseFromUrl = parseInt(new URLSearchParams(location.search).get("case"), 10);
if (caseFromUrl >= 1 && caseFromUrl <= METRICS.demo_cases.length) loadCase(caseFromUrl - 1);
LHModel.load().then(renderSafety).catch(err => {
  console.error(err);
  $("#verdict").innerHTML = verdictHtml("!", "AI model failed to load",
    "Open this page through a web server (see README), not by double-clicking the file.");
});
