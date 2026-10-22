// Guided recording: walks the health worker through the four listening sites,
// shows the live sound with a heartbeat check, records 15 seconds per site and
// hands each recording to App for analysis.

const Guided = (() => {
  const SITES = ["AV", "PV", "TV", "MV"];
  const RECORD_SECONDS = 15;
  const GET_READY_SECONDS = 3;
  const LIVE_SECONDS = 6;

  // Runs on the audio thread and forwards raw microphone samples.
  const WORKLET = `
    class Tap extends AudioWorkletProcessor {
      process(inputs) {
        if (inputs[0] && inputs[0][0]) this.port.postMessage(inputs[0][0].slice(0));
        return true;
      }
    }
    registerProcessor("tap", Tap);`;

  let ctx = null, stream = null, node = null;
  let live = [];            // last LIVE_SECONDS of samples, for the quality check
  let take = null;          // samples of the recording in progress
  let queue = [];           // sites still to record
  let site = null, phase = "idle", phaseEnds = 0, total = 0, step = 0;
  let timer = null, liveCheck = null, liveState = { level: 0, beat: null };

  // ---------------------------------------------------------- audio

  async function openMic() {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    node = new AudioWorkletNode(ctx, "tap");
    node.port.onmessage = (e) => onSamples(e.data);
    ctx.createMediaStreamSource(stream).connect(node);
  }

  function closeMic() {
    clearInterval(timer);
    clearInterval(liveCheck);
    if (node) node.port.onmessage = null;
    if (stream) stream.getTracks().forEach(tr => tr.stop());
    if (ctx) ctx.close();
    ctx = stream = node = null;
    live = [];
    take = null;
  }

  function onSamples(chunk) {
    live.push(chunk);
    let buffered = live.reduce((n, c) => n + c.length, 0);
    while (buffered - live[0].length > LIVE_SECONDS * ctx.sampleRate) buffered -= live.shift().length;
    if (phase === "recording") take.push(chunk);
  }

  function concat(chunks) {
    const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  // Quick resampling for the live check only: average blocks, then pick samples.
  function toModelRate(x, rate, target) {
    const ratio = rate / target;
    const width = Math.max(1, Math.round(ratio));
    const out = new Float64Array(Math.floor(x.length / ratio));
    for (let i = 0; i < out.length; i++) {
      const start = Math.floor(i * ratio);
      let sum = 0, n = 0;
      for (let j = start; j < Math.min(x.length, start + width); j++) { sum += x[j]; n++; }
      out[i] = n ? sum / n : 0;
    }
    return out;
  }

  // 16-bit PCM WAV, so the normal upload path (and playback) can use the take.
  function encodeWav(samples, rate) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const str = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
    str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
    }
    return buf;
  }

  // ---------------------------------------------------------- live quality check

  function checkLive() {
    if (!ctx || !live.length) return;
    const x = concat(live);
    Draw.wave($("#live-wave"), x.subarray(Math.max(0, x.length - 3 * ctx.sampleRate)));
    const recent = x.subarray(Math.max(0, x.length - 0.3 * ctx.sampleRate));
    let sq = 0;
    for (let i = 0; i < recent.length; i++) sq += recent[i] * recent[i];
    liveState.level = Math.sqrt(sq / Math.max(1, recent.length));
    if (x.length >= 4 * ctx.sampleRate && LHModel.config) {
      const f = LHModel.config.features;
      const spec = LHFeatures.logMel(toModelRate(x, ctx.sampleRate, f.sample_rate), f);
      const m = LHFeatures.beatMeasures(spec, LHModel.beatCheck);
      liveState.beat = m.rhythm >= LHModel.beatCheck.rhythm_min && m.depth >= LHModel.beatCheck.depth_min;
    }
    renderLive();
  }

  function renderLive() {
    const chip = $("#beat-chip");
    let cls = "warn", text = t("live.searching");
    if (liveState.level < 0.002) { cls = "bad"; text = t("live.quiet"); }
    else if (liveState.beat) { cls = "ok"; text = t("live.beat"); }
    chip.className = "chip-status " + cls;
    chip.textContent = text;
    const db = 20 * Math.log10(liveState.level + 1e-9);
    $("#level-meter span").style.width = Math.max(0, Math.min(100, (db + 70) / 60 * 100)) + "%";
  }

  // ---------------------------------------------------------- steps

  function renderStep() {
    const left = Math.max(0, Math.ceil((phaseEnds - performance.now()) / 1000));
    $("#guide-step").textContent = total > 1
      ? t("guide.step", { n: step, total, site: t("site." + site) })
      : t("guide.single", { site: t("site." + site) });
    $("#guide-instruction").textContent = t("guide.place." + site);
    $("#countdown").textContent = phase === "ready" ? t("guide.ready", { s: left }) : t("guide.recording", { s: left });
    $("#countdown").className = "countdown " + phase;
  }

  function startSite(next, again = false) {
    if (!again) step += 1;
    site = next;
    App.highlightSite(site);
    phase = "ready";
    phaseEnds = performance.now() + GET_READY_SECONDS * 1000;
    renderStep();
  }

  function tick() {
    if (performance.now() < phaseEnds) return renderStep();
    if (phase === "ready") {
      phase = "recording";
      take = [];
      phaseEnds = performance.now() + RECORD_SECONDS * 1000;
      return renderStep();
    }
    if (phase === "recording") finishSite();
  }

  function finishSite() {
    const samples = concat(take);
    take = null;
    App.replaceSite(site);
    App.addRecordings([{ buffer: encodeWav(samples, ctx.sampleRate), site, name: t("guide.takeName", { site }) }]);
    nextSite();
  }

  function nextSite() {
    if (queue.length) return startSite(queue.shift());
    stop(t("guide.done"));
  }

  async function begin(sites) {
    const status = $("#record-status");
    try {
      await LHModel.load();
      await openMic();
    } catch (err) {
      console.error(err);
      status.textContent = t("guide.denied");
      closeMic();
      return;
    }
    App.leaveDemoCase();
    status.textContent = "";
    queue = sites.slice(1);
    total = sites.length;
    step = 0;
    $("#guide-idle").hidden = true;
    $("#guide-live").hidden = false;
    liveState = { level: 0, beat: null };
    startSite(sites[0]);
    timer = setInterval(tick, 200);
    liveCheck = setInterval(checkLive, 1000);
  }

  function stop(message) {
    phase = "idle";
    closeMic();
    $("#guide-idle").hidden = false;
    $("#guide-live").hidden = true;
    $("#record-status").textContent = message || "";
    App.highlightSite(null);
  }

  function init() {
    $("#guide-start").addEventListener("click", () => begin(SITES));
    $("#record-one").addEventListener("click", () => begin([$("#record-site").value]));
    $("#record-site").addEventListener("change", () => App.highlightSite($("#record-site").value));
    $("#guide-skip").addEventListener("click", () => { take = null; nextSite(); });
    $("#guide-redo").addEventListener("click", () => startSite(site, true));
    $("#guide-stop").addEventListener("click", () => stop(t("guide.stopped")));
  }

  return { init, relabel: () => { if (phase !== "idle") { renderStep(); renderLive(); } }, encodeWav };
})();
