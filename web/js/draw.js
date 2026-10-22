// Drawing helpers shared by the screening, guided recording and quiz views.

const Draw = (() => {
  const COLOR_STOPS = [[13, 8, 35], [62, 24, 104], [158, 47, 127], [234, 105, 84], [252, 216, 141]];

  function colorAt(v) {
    const x = Math.min(0.9999, Math.max(0, v)) * (COLOR_STOPS.length - 1);
    const i = Math.floor(x), f = x - i;
    return COLOR_STOPS[i].map((c, k) => Math.round(c + (COLOR_STOPS[i + 1][k] - c) * f));
  }

  // Low pitches at the bottom, time from left to right.
  function spectrogram(canvas, spec) {
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

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name);
  }

  // One bar per 4-second window: murmur likelihood, dashed line at the referral threshold.
  function timeline(canvas, result, threshold, caption) {
    const w = canvas.clientWidth || 600, h = 56;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const x = (sec) => (sec / result.duration) * w;
    const y = (p) => h - 4 - p * (h - 14);

    ctx.strokeStyle = cssVar("--border");
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(0, y(threshold)); ctx.lineTo(w, y(threshold)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cssVar("--muted");
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText(caption, 4, 10);

    const span = Math.min(4, result.duration);
    result.windows.forEach(win => {
      const p = win.probs[1];
      ctx.fillStyle = p >= threshold ? cssVar("--bad") : cssVar("--ok");
      ctx.globalAlpha = 0.25 + 0.6 * p;
      const left = x(win.startSec + span / 2 - 1), right = x(Math.min(result.duration, win.startSec + span / 2 + 1));
      ctx.fillRect(left, y(p), Math.max(2, right - left - 1), h - 4 - y(p));
    });
    ctx.globalAlpha = 1;
  }

  // Play button with a moving cursor over the spectrogram.
  function playback(el, duration) {
    const audio = el.querySelector("audio");
    const btn = el.querySelector(".play");
    const head = el.querySelector(".playhead");
    let raf = null;
    const tick = () => {
      if (head) head.style.left = (100 * audio.currentTime / duration) + "%";
      if (!audio.paused) raf = requestAnimationFrame(tick);
    };
    btn.addEventListener("click", () => {
      document.querySelectorAll("audio").forEach(a => { if (a !== audio) a.pause(); });
      if (audio.paused) audio.play(); else audio.pause();
    });
    audio.addEventListener("play", () => { btn.textContent = "❚❚"; if (head) head.style.display = "block"; tick(); });
    audio.addEventListener("pause", () => { btn.textContent = "▶"; cancelAnimationFrame(raf); });
    audio.addEventListener("ended", () => { btn.textContent = "▶"; if (head) head.style.display = "none"; });
  }

  // Live microphone view: min/max envelope of the last few seconds.
  function wave(canvas, samples) {
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 70;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = cssVar("--accent");
    let peak = 1e-4;
    for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]));
    const per = Math.max(1, Math.floor(samples.length / w));
    for (let px = 0; px < w; px++) {
      let lo = 0, hi = 0;
      for (let i = px * per; i < Math.min(samples.length, (px + 1) * per); i++) {
        lo = Math.min(lo, samples[i]); hi = Math.max(hi, samples[i]);
      }
      const top = h / 2 - (hi / peak) * (h / 2 - 2), bottom = h / 2 - (lo / peak) * (h / 2 - 2);
      ctx.fillRect(px, top, 1, Math.max(1, bottom - top));
    }
  }

  return { spectrogram, timeline, playback, wave };
})();
