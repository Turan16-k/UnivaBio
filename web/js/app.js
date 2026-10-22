// Screening view: real cases, uploads, results per child and per valve.
// Recording from a microphone lives in guided.js.

const $ = (sel) => document.querySelector(sel);
const pct = (x) => (I18N.lang === "tr" ? "%" + Math.round(x * 100) : Math.round(x * 100) + "%");
const t = (key, vars) => I18N.t(key, vars);
const LABEL_CLASS = ["ok", "bad", "warn"];

const App = (() => {
  let recordings = [];   // { id, site, name, url, result }
  let currentCase = null;
  let nextId = 1;

  const siteName = (site) => t("site." + site);

  // Adds all recordings first (so the result waits for every one of them),
  // then analyses them one by one. Results for recordings that were cleared
  // in the meantime (e.g. the user picked another case) are thrown away.
  async function addRecordings(items) {
    const recs = items.map(({ buffer, site, name }) => ({
      id: nextId++, site, name, buffer, url: URL.createObjectURL(new Blob([buffer])), result: null,
    }));
    recordings.push(...recs);
    renderAll();
    for (const rec of recs) {
      if (!recordings.includes(rec)) return;
      try {
        const samples = await LHFeatures.decodeToRate(rec.buffer, LHModel.config?.features.sample_rate || 4000);
        const result = await LHModel.analyzeRecording(samples);
        if (!recordings.includes(rec)) return;
        rec.result = result;
      } catch (err) {
        console.error(err);
        rec.error = true;
      }
      delete rec.buffer;
      renderAll();
    }
  }

  function replaceSite(site) {
    recordings.filter(r => r.site === site).forEach(r => URL.revokeObjectURL(r.url));
    recordings = recordings.filter(r => r.site !== site);
  }

  function clearRecordings() {
    recordings.forEach(r => URL.revokeObjectURL(r.url));
    recordings = [];
    currentCase = null;
    document.querySelectorAll("[data-case]").forEach(b => b.classList.remove("active"));
    renderAll();
  }

  function leaveDemoCase() {
    if (currentCase) clearRecordings();
  }

  // ---------------------------------------------------------- demo cases and uploads

  function renderDemoCases() {
    $("#demo-cases").innerHTML = METRICS.demo_cases.map((c, i) => `
      <button class="case${currentCase === c ? " active" : ""}" data-case="${i}">
        <span class="case-title">${t("demo.case", { n: i + 1 })}</span>
        <span class="case-meta">${t("age." + c.age)}, ${t("sex." + c.sex)} · ${t("demo.recs", { n: c.recordings.length })}</span>
      </button>`).join("");
    document.querySelectorAll("[data-case]").forEach(btn => btn.addEventListener("click", () => loadCase(+btn.dataset.case)));
  }

  async function loadCase(index) {
    clearRecordings();
    document.querySelectorAll("[data-case]").forEach(b => b.classList.toggle("active", +b.dataset.case === index));
    currentCase = METRICS.demo_cases[index];
    const caseRef = currentCase;
    if (window.innerWidth < 980) $("#verdict-card").scrollIntoView({ behavior: "smooth", block: "start" });
    const items = await Promise.all(caseRef.recordings.map(async r => ({
      buffer: await (await fetch("samples/" + r.file)).arrayBuffer(), site: r.location, name: r.file,
    })));
    if (currentCase !== caseRef) return;
    await addRecordings(items);
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
      leaveDemoCase();
      const order = ["AV", "PV", "TV", "MV"];
      const used = new Set(recordings.map(r => r.site));
      const items = [];
      for (const file of files) {
        const site = siteFromName(file.name) || order.find(s => !used.has(s)) || "Phc";
        used.add(site);
        items.push({ buffer: await file.arrayBuffer(), site, name: file.name });
      }
      await addRecordings(items);
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

  // ---------------------------------------------------------- results

  function decision() {
    const done = recordings.filter(r => r.result);
    if (!done.length || done.length < recordings.filter(r => !r.error).length) return null;
    const d = LHModel.decide(done.map(r => r.result));
    return { ...d, done, sites: done.map((r, i) => ({ site: r.site, label: d.labels[i], present: r.result.probs[1] })) };
  }

  function renderAll() {
    renderChest();
    renderVerdict();
    renderSiteResults();
  }

  function siteLabel(site) {
    const here = recordings.filter(r => r.site === site);
    const done = here.filter(r => r.result);
    if (!done.length) return here.some(r => !r.error) ? "busy" : null;
    const labels = done.map(r => LHModel.recordingLabel(r.result));
    if (labels.includes(LHModel.PRESENT)) return LABEL_CLASS[1];
    if (labels.includes(LHModel.UNKNOWN)) return LABEL_CLASS[2];
    return LABEL_CLASS[0];
  }

  function renderChest() {
    document.querySelectorAll(".site").forEach(g => {
      const focus = g.classList.contains("focus") ? " focus" : "";
      g.setAttribute("class", "site " + (siteLabel(g.dataset.site) || "") + focus);
    });
  }

  function highlightSite(site) {
    document.querySelectorAll(".site").forEach(g => g.classList.toggle("focus", g.dataset.site === site));
  }

  function verdictHtml(icon, title, text) {
    return `<div class="verdict-icon" aria-hidden="true">${icon}</div>
      <div><div class="verdict-title">${title}</div><div class="verdict-text">${text}</div></div>`;
  }

  function renderVerdict() {
    const box = $("#verdict");
    const truth = $("#truth");
    const done = recordings.filter(r => r.result);
    const pending = recordings.filter(r => !r.result && !r.error).length;
    truth.hidden = true;
    $("#verdict-actions").hidden = !done.length || pending > 0;

    if (!recordings.length) {
      box.className = "verdict empty";
      box.innerHTML = verdictHtml("♡", t("verdict.empty.title"), t("verdict.empty.text"));
      return;
    }
    if (!done.length) {
      box.className = "verdict empty";
      box.innerHTML = verdictHtml("…", t("verdict.busy.title"), t("verdict.busy.text"));
      return;
    }

    const d = LHModel.decide(done.map(r => r.result));
    const trusted = done.filter(r => r.result.beat.ok);
    const strongest = trusted.length ? trusted.reduce((a, b) => (b.result.probs[1] > a.result.probs[1] ? b : a)) : done[0];
    const unclear = [...new Set(done.filter((r, i) => d.labels[i] === LHModel.UNKNOWN).map(r => r.site))];
    const noBeat = [...new Set(done.filter(r => !r.result.beat.ok).map(r => r.site))];
    const more = pending ? " " + t("verdict.pending", { n: pending }) : "";

    if (d.label === LHModel.PRESENT) {
      box.className = "verdict bad";
      box.innerHTML = verdictHtml("!", t("verdict.refer.title"),
        t("verdict.refer.text", { site: siteName(strongest.site).toLocaleLowerCase(I18N.locale), p: pct(d.present) }) + more);
    } else if (d.label === LHModel.UNKNOWN && noBeat.length && done.every(r => !r.result.beat.ok)) {
      box.className = "verdict warn";
      box.innerHTML = verdictHtml("?", t("verdict.nobeat.title"), t("verdict.nobeat.text") + more);
    } else if (d.label === LHModel.UNKNOWN) {
      box.className = "verdict warn";
      const why = noBeat.length ? " " + t("verdict.unclear.nobeat", { sites: noBeat.join(", ") }) : "";
      box.innerHTML = verdictHtml("?", t("verdict.unclear.title"),
        t("verdict.unclear.text", { sites: unclear.join(", ") }) + why + more);
    } else {
      box.className = "verdict ok";
      box.innerHTML = verdictHtml("✓", t("verdict.clear.title"),
        t("verdict.clear.text", { p: pct(d.present), th: pct(d.thresholds.present) }) + more);
    }

    if (currentCase && !pending) {
      const c = currentCase;
      const truthLabel = ["Absent", "Present", "Unknown"].indexOf(c.truth);
      const agree = truthLabel === d.label;
      let details;
      if (c.truth === "Present") {
        details = t("truth.murmur", { sites: c.murmur_locations.join(", ") });
        if (c.timing !== "nan") details += " · " + t("timing." + c.timing);
        if (c.grade !== "nan") details += ", " + t("truth.grade", { g: c.grade });
      } else {
        details = c.truth === "Absent" ? t("truth.none") : t("truth.unknown");
      }
      truth.hidden = false;
      truth.className = "truth " + (agree ? "agree" : "disagree");
      truth.innerHTML = `<b>${t("truth.label")}</b> ${details}. <span>${agree ? t("truth.agree") : t("truth.disagree")}</span>`;
    }
  }

  function renderSiteResults() {
    const container = $("#site-results");
    const order = { AV: 0, PV: 1, TV: 2, MV: 3, Phc: 4 };
    const sorted = [...recordings].sort((a, b) => order[a.site] - order[b.site] || a.id - b.id);
    const existing = new Map([...container.children].map(el => [+el.dataset.id, el]));
    sorted.forEach(rec => {
      let el = existing.get(rec.id);
      const state = rec.error ? "error" : rec.result ? "done" : "busy";
      if (!el || el.dataset.state !== state || el.dataset.lang !== I18N.lang) {
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
    el.dataset.lang = I18N.lang;
    const title = `<div class="site-head"><h3>${siteName(rec.site)} <span class="site-code">${rec.site}</span></h3>`;

    if (state === "busy") {
      el.innerHTML = `${title}<span class="badge">${t("card.busy")}</span></div><div class="skeleton"></div>`;
      return el;
    }
    if (state === "error") {
      el.innerHTML = `${title}<span class="badge warn">${t("card.errorBadge")}</span></div><p class="small">${t("card.error")}</p>`;
      return el;
    }

    const r = rec.result;
    const th = LHModel.config.thresholds;
    const label = LHModel.recordingLabel(r);
    const badge = r.beat.ok ? t("label." + label) : t("card.nobeat");
    el.innerHTML = `${title}<span class="badge ${LABEL_CLASS[label]}">${badge}</span></div>
      <div class="spec-wrap">
        <canvas class="spec" aria-label="${t("card.specAria")}"></canvas>
        <div class="playhead"></div>
        <span class="axis-label top">1000 Hz</span><span class="axis-label bottom">25 Hz</span>
      </div>
      <canvas class="timeline" aria-label="${t("card.timelineAria")}"></canvas>
      <div class="site-foot">
        <button class="play" aria-label="${t("card.play")}">▶</button>
        <span class="small muted">${r.duration.toFixed(1)} s · ${r.beat.ok ? t("card.beatOk") : t("card.beatNo")} ·
          ${t("card.murmur")} <b>${pct(r.probs[1])}</b> · ${t("card.unclear")} ${pct(r.probs[2])}</span>
      </div>
      <audio preload="none" src="${rec.url}"></audio>`;
    Draw.spectrogram(el.querySelector(".spec"), r.spec);
    requestAnimationFrame(() => Draw.timeline(el.querySelector(".timeline"), r, th.present, t("card.timelineCaption")));
    Draw.playback(el, r.duration);
    return el;
  }

  // ---------------------------------------------------------- setup

  function init() {
    setupUpload();
    renderDemoCases();
    renderAll();
    $("#print-btn").addEventListener("click", () => {
      $("#print-meta").textContent = t("print.meta", { date: new Date().toLocaleString(I18N.locale), n: recordings.length });
      window.print();
    });
    $("#clear-btn").addEventListener("click", clearRecordings);
    document.querySelectorAll(".site").forEach(g => g.addEventListener("click", () => {
      $("#record-site").value = g.dataset.site;
      highlightSite(g.dataset.site);
    }));
    const caseFromUrl = parseInt(new URLSearchParams(location.search).get("case"), 10);
    if (caseFromUrl >= 1 && caseFromUrl <= METRICS.demo_cases.length) {
      document.querySelector('[data-source="demo"]').click();
      loadCase(caseFromUrl - 1);
    }
  }

  function relabel() {
    renderDemoCases();
    renderAll();
  }

  return {
    init, relabel, addRecordings, replaceSite, clearRecordings, leaveDemoCase, highlightSite, decision,
    get isDemo() { return currentCase !== null; },
  };
})();
