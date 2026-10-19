// Helper for tests/test_pipeline.py: prints the JavaScript log-mel of a WAV file as JSON.
// Usage: node tests/js_logmel.cjs <file.wav> <config.json>
const fs = require("fs");
const path = require("path");
const F = require(path.join(__dirname, "..", "web", "js", "features.js"));

const [wavPath, configPath] = process.argv.slice(2);
const buf = fs.readFileSync(wavPath);
const wav = F.parseWav(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const cfg = JSON.parse(fs.readFileSync(configPath, "utf8")).features;
const spec = F.logMel(wav.samples, cfg);
const starts = F.windowStarts(spec.nFrames, cfg);
const check = JSON.parse(fs.readFileSync(path.join(path.dirname(configPath), "heartbeat_check.json"), "utf8"));
const beat = F.beatMeasures(spec, check);
process.stdout.write(JSON.stringify({ nFrames: spec.nFrames, data: Array.from(spec.data), starts, beat }));
