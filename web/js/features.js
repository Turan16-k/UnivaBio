// Audio loading and log-mel spectrogram, a line-by-line port of ml/features.py.
// tests/parity.test.mjs checks that both versions produce the same numbers.

(function (root) {
  "use strict";

  // Reads a 16-bit PCM WAV file (the CirCor format). Returns null for anything else,
  // so the caller can fall back to the browser's decoder.
  function parseWav(buffer) {
    const view = new DataView(buffer);
    const tag = (o) => String.fromCharCode(view.getUint8(o), view.getUint8(o + 1), view.getUint8(o + 2), view.getUint8(o + 3));
    if (buffer.byteLength < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
    let offset = 12, fmt = null;
    while (offset + 8 <= buffer.byteLength) {
      const id = tag(offset), size = view.getUint32(offset + 4, true);
      if (id === "fmt ") {
        fmt = {
          format: view.getUint16(offset + 8, true),
          channels: view.getUint16(offset + 10, true),
          sampleRate: view.getUint32(offset + 12, true),
          bits: view.getUint16(offset + 22, true),
        };
      } else if (id === "data" && fmt) {
        if (fmt.format !== 1 || fmt.bits !== 16) return null;
        const n = Math.floor(Math.min(size, buffer.byteLength - offset - 8) / (2 * fmt.channels));
        const samples = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let c = 0; c < fmt.channels; c++) {
            sum += view.getInt16(offset + 8 + 2 * (i * fmt.channels + c), true) / 32768;
          }
          samples[i] = sum / fmt.channels;
        }
        return { sampleRate: fmt.sampleRate, samples };
      }
      offset += 8 + size + (size % 2);
    }
    return null;
  }

  // Converts any audio the browser can decode (mic recording, mp3, 44.1 kHz wav...)
  // to mono at the model's sample rate.
  async function decodeToRate(buffer, sampleRate) {
    const wav = parseWav(buffer.slice(0));
    if (wav && wav.sampleRate === sampleRate) return wav.samples;
    const decoded = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(buffer.slice(0));
    try {
      return await resample(decoded, sampleRate);
    } catch {
      // Some browsers refuse very low sample rates: render at 2x, then average sample pairs.
      const double = await resample(decoded, sampleRate * 2);
      const out = new Float64Array(Math.floor(double.length / 2));
      for (let i = 0; i < out.length; i++) out[i] = (double[2 * i] + double[2 * i + 1]) / 2;
      return out;
    }
  }

  async function resample(decoded, sampleRate) {
    const length = Math.ceil(decoded.duration * sampleRate);
    const offline = new OfflineAudioContext(1, length, sampleRate);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    return Float64Array.from(rendered.getChannelData(0));
  }

  function normalize(x) {
    let mean = 0;
    for (let i = 0; i < x.length; i++) mean += x[i];
    mean /= x.length;
    let variance = 0;
    for (let i = 0; i < x.length; i++) variance += (x[i] - mean) ** 2;
    const std = Math.sqrt(variance / x.length) + 1e-8;
    const out = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = (x[i] - mean) / std;
    return out;
  }

  // In-place iterative radix-2 FFT on separate real/imaginary arrays.
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const angle = (-2 * Math.PI) / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < len / 2; k++) {
          const wr = Math.cos(angle * k), wi = Math.sin(angle * k);
          const ar = re[i + k + len / 2], ai = im[i + k + len / 2];
          const tr = ar * wr - ai * wi, ti = ar * wi + ai * wr;
          re[i + k + len / 2] = re[i + k] - tr;
          im[i + k + len / 2] = im[i + k] - ti;
          re[i + k] += tr;
          im[i + k] += ti;
        }
      }
    }
  }

  // Returns { data: Float32Array(nMels * nFrames) in row-major (mel, frame) order, nFrames }.
  function logMel(samples, cfg) {
    const { n_fft: nFft, hop, n_mels: nMels, eps, mel_filterbank: fb } = cfg;
    let x = normalize(samples);
    if (x.length < nFft) {
      const padded = new Float64Array(nFft);
      padded.set(x);
      x = padded;
    }
    const nFrames = 1 + Math.floor((x.length - nFft) / hop);
    const nBins = nFft / 2 + 1;
    const hann = new Float64Array(nFft);
    for (let i = 0; i < nFft; i++) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft);

    const out = new Float32Array(nMels * nFrames);
    const re = new Float64Array(nFft), im = new Float64Array(nFft), power = new Float64Array(nBins);
    for (let f = 0; f < nFrames; f++) {
      for (let i = 0; i < nFft; i++) {
        re[i] = x[f * hop + i] * hann[i];
        im[i] = 0;
      }
      fft(re, im);
      for (let b = 0; b < nBins; b++) power[b] = re[b] * re[b] + im[b] * im[b];
      for (let m = 0; m < nMels; m++) {
        let sum = 0;
        const row = fb[m];
        for (let b = 0; b < nBins; b++) if (row[b] !== 0) sum += row[b] * power[b];
        out[m * nFrames + f] = Math.log10(sum + eps);
      }
    }
    return { data: out, nFrames };
  }

  function windowStarts(nFrames, cfg) {
    if (nFrames <= cfg.window_frames) return [0];
    const starts = [];
    for (let s = 0; s + cfg.window_frames <= nFrames; s += cfg.window_step) starts.push(s);
    return starts;
  }

  // One window, shape (nMels, windowFrames). Short recordings are repeated to fill it.
  function cutWindow(spec, start, cfg) {
    const { n_mels: nMels, window_frames: w } = cfg;
    const out = new Float32Array(nMels * w);
    for (let m = 0; m < nMels; m++) {
      for (let t = 0; t < w; t++) {
        out[m * w + t] = spec.data[m * spec.nFrames + ((start + t) % spec.nFrames)];
      }
    }
    return out;
  }

  // Port of beat_measures in ml/heartbeat_check.py: is there a heartbeat rhythm at all?
  function beatMeasures(spec, check) {
    const n = spec.nFrames;
    const env = new Float64Array(n);
    for (let f = 0; f < n; f++) {
      let sum = 0;
      for (let m = 0; m < check.low_bands; m++) sum += spec.data[m * n + f];
      env[f] = sum / check.low_bands;
    }
    let mean = 0;
    for (let f = 0; f < n; f++) mean += env[f];
    mean /= n;
    let denom = 0;
    for (let f = 0; f < n; f++) { env[f] -= mean; denom += env[f] * env[f]; }
    const depth = Math.sqrt(denom / Math.max(1, n));
    if (denom <= 0 || n <= check.max_lag + 1) return { rhythm: 0, depth };
    let rhythm = 0;
    for (let lag = check.min_lag; lag <= check.max_lag; lag++) {
      let r = 0;
      for (let f = 0; f + lag < n; f++) r += env[f] * env[f + lag];
      rhythm = Math.max(rhythm, r / denom);
    }
    return { rhythm, depth };
  }

  const api = { parseWav, decodeToRate, normalize, fft, logMel, windowStarts, cutWindow, beatMeasures };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LHFeatures = api;
})(typeof self !== "undefined" ? self : this);
