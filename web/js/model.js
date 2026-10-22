// Runs the exported CNN in the browser with ONNX Runtime Web and applies the
// same decision rule as ml/train.py (decide / patient_probs), plus the
// heartbeat safety check from ml/heartbeat_check.py.

(function (root) {
  "use strict";

  const ABSENT = 0, PRESENT = 1, UNKNOWN = 2;
  let session = null;
  let config = null;
  let beatCheck = null;
  let loading = null;

  function load() {
    loading = loading || (async () => {
      [config, beatCheck] = await Promise.all([
        fetch("model/config.json").then(r => r.json()),
        fetch("model/heartbeat_check.json").then(r => r.json()),
      ]);
      ort.env.wasm.wasmPaths = new URL("vendor/ort/", document.baseURI).href;
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create("model/murmur_cnn.onnx", { executionProviders: ["wasm"] });
    })();
    return loading;
  }

  // ONNX Runtime can only run one inference at a time, so calls wait in line.
  let queue = Promise.resolve();
  function analyzeRecording(samples) {
    const job = queue.then(() => analyzeNow(samples));
    queue = job.catch(() => {});
    return job;
  }

  // Analyses one recording: spectrogram, heartbeat check, 4-second windows, CNN, average.
  async function analyzeNow(samples) {
    await load();
    const f = config.features;
    const spec = LHFeatures.logMel(samples, f);
    const m = LHFeatures.beatMeasures(spec, beatCheck);
    const beat = { ...m, ok: m.rhythm >= beatCheck.rhythm_min && m.depth >= beatCheck.depth_min };

    const starts = LHFeatures.windowStarts(spec.nFrames, f);
    const size = f.n_mels * f.window_frames;
    const batch = new Float32Array(starts.length * size);
    starts.forEach((s, i) => batch.set(LHFeatures.cutWindow(spec, s, f), i * size));
    const input = new ort.Tensor("float32", batch, [starts.length, 1, f.n_mels, f.window_frames]);
    const { probs } = await session.run({ logmel: input });
    const windows = starts.map((s, i) => ({
      startSec: (s * f.hop) / f.sample_rate,
      probs: Array.from(probs.data.slice(i * 3, i * 3 + 3)),
    }));
    const mean = [0, 1, 2].map(c => windows.reduce((sum, w) => sum + w.probs[c], 0) / windows.length);
    return { spec, windows, probs: mean, beat, duration: samples.length / f.sample_rate };
  }

  // Label for one recording. A recording without a heartbeat rhythm is never
  // trusted as "murmur" or "no murmur": it must be recorded again.
  function recordingLabel(r) {
    const t = config.thresholds;
    if (!r.beat.ok) return UNKNOWN;
    if (r.probs[PRESENT] >= t.present) return PRESENT;
    if (r.probs[UNKNOWN] >= t.unknown) return UNKNOWN;
    return ABSENT;
  }

  // Child-level decision: Present if any valve is Present, else Unknown if
  // any recording is unclear, else Absent.
  function decide(results) {
    const labels = results.map(recordingLabel);
    const trusted = results.filter(r => r.beat.ok);
    const present = trusted.length ? Math.max(...trusted.map(r => r.probs[PRESENT])) : 0;
    let label = ABSENT;
    if (labels.includes(UNKNOWN)) label = UNKNOWN;
    if (labels.includes(PRESENT)) label = PRESENT;
    return { label, labels, present, noBeat: results.filter(r => !r.beat.ok).length, thresholds: config.thresholds };
  }

  root.LHModel = {
    load, analyzeRecording, decide, recordingLabel, ABSENT, PRESENT, UNKNOWN,
    get config() { return config; },
    get beatCheck() { return beatCheck; },
  };
})(self);
