"""Run with:  python -m pytest tests"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

import features as F  # noqa: E402
from heartbeat_check import beat_measures  # noqa: E402
from train import ABSENT, PRESENT, UNKNOWN, decide, weighted_accuracy  # noqa: E402

SAMPLES = sorted((ROOT / "web" / "samples").glob("*.wav"))
CONFIG = ROOT / "web" / "model" / "config.json"
CHECK = ROOT / "web" / "model" / "heartbeat_check.json"


def test_mel_filters_cover_range_in_order():
    fb = F.MEL_FB
    assert fb.shape == (F.N_MELS, F.N_FFT // 2 + 1)
    peaks = fb.argmax(axis=1)
    assert np.all(np.diff(peaks) >= 0), "filters should go from low to high pitch"
    assert np.all(fb.max(axis=1) > 0), "every filter should catch some frequency"


def test_pure_tone_lands_in_matching_band():
    t = np.arange(F.SAMPLE_RATE * 2) / F.SAMPLE_RATE
    for freq in (60, 200, 600):
        spec = F.log_mel(np.sin(2 * np.pi * freq * t))
        band = spec.mean(axis=1).argmax()
        centers = F.mel_to_hz(np.linspace(F.hz_to_mel(F.F_MIN), F.hz_to_mel(F.F_MAX), F.N_MELS + 2))[1:-1]
        assert abs(centers[band] - freq) < 40, f"{freq} Hz tone went to band centred at {centers[band]:.0f} Hz"


def test_loudness_does_not_change_spectrogram():
    rng = np.random.default_rng(0)
    x = rng.normal(size=8000)
    assert np.allclose(F.log_mel(x), F.log_mel(x * 7.5), atol=1e-4)


def test_short_recording_is_repeated_to_fill_window():
    spec = np.arange(F.N_MELS * 100, dtype=np.float32).reshape(F.N_MELS, 100)
    w = F.cut_window(spec, 0)
    assert w.shape == (F.N_MELS, F.WINDOW_FRAMES)
    assert np.array_equal(w[:, 100:200], spec)


def test_weighted_accuracy_matches_challenge_definition():
    # 2 present (1 correct), 1 unknown (correct), 3 absent (2 correct)
    y_true = [PRESENT, PRESENT, UNKNOWN, ABSENT, ABSENT, ABSENT]
    y_pred = [PRESENT, ABSENT, UNKNOWN, ABSENT, ABSENT, PRESENT]
    expected = (5 * 1 + 3 * 1 + 1 * 2) / (5 * 2 + 3 * 1 + 1 * 3)
    assert weighted_accuracy(y_true, y_pred) == pytest.approx(expected)


def test_decision_rule_order():
    probs = np.array([[0.9, 0.9], [0.1, 0.9], [0.1, 0.1]])  # (present, unknown) per child
    assert list(decide(probs, 0.5, 0.5)) == [PRESENT, UNKNOWN, ABSENT]


@pytest.mark.skipif(not SAMPLES or not CONFIG.exists(), reason="run ml/train.py first")
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_javascript_features_match_python():
    for wav in SAMPLES[:3]:
        out = subprocess.run(["node", str(ROOT / "tests" / "js_logmel.cjs"), str(wav), str(CONFIG)],
                             capture_output=True, text=True, check=True)
        js = json.loads(out.stdout)
        py = F.log_mel(F.load_wav(wav))
        assert js["nFrames"] == py.shape[1]
        assert js["starts"] == F.window_starts(py.shape[1])
        diff = np.abs(np.array(js["data"]).reshape(py.shape) - py).max()
        assert diff < 1e-3, f"{wav.name}: max difference {diff}"
        rhythm, depth = beat_measures(py)
        assert js["beat"]["rhythm"] == pytest.approx(rhythm, abs=1e-3)
        assert js["beat"]["depth"] == pytest.approx(depth, abs=1e-3)


@pytest.mark.skipif(not CHECK.exists(), reason="run ml/heartbeat_check.py first")
def test_heartbeat_check_rejects_non_heart_sounds_and_accepts_demo_cases():
    limits = json.loads(CHECK.read_text())
    passes = lambda m: m[0] >= limits["rhythm_min"] and m[1] >= limits["depth_min"]
    rng = np.random.default_rng(123)
    t = np.arange(F.SAMPLE_RATE * 10) / F.SAMPLE_RATE
    assert not passes(beat_measures(F.log_mel(rng.normal(size=len(t)))))
    assert not passes(beat_measures(F.log_mel(np.sin(2 * np.pi * 440 * t))))
    accepted = [passes(beat_measures(F.log_mel(F.load_wav(w)))) for w in SAMPLES]
    assert np.mean(accepted) >= 0.9


@pytest.mark.skipif(not CONFIG.exists(), reason="run ml/train.py first")
def test_exported_model_outputs_probabilities():
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(ROOT / "web" / "model" / "murmur_cnn.onnx"))
    x = np.random.default_rng(1).normal(-1, 1, (3, 1, F.N_MELS, F.WINDOW_FRAMES)).astype(np.float32)
    probs = session.run(None, {"logmel": x})[0]
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)


def test_add_noise_hits_requested_snr():
    import evaluation as E
    rng = np.random.default_rng(3)
    x = np.sin(2 * np.pi * 100 * np.arange(40000) / F.SAMPLE_RATE)
    for snr in (20, 10, 5):
        noise = E.add_noise(x, snr, rng) - x
        measured = 10 * np.log10(np.mean(x ** 2) / np.mean(noise ** 2))
        assert measured == pytest.approx(snr, abs=0.3)


def test_augmented_training_window_has_model_shape():
    import evaluation as E
    rng = np.random.default_rng(4)
    audio = (rng.normal(size=F.SAMPLE_RATE * 9) * 3000).astype(np.int16)
    short = audio[:F.SAMPLE_RATE * 2]
    for a in (audio, short):
        for _ in range(5):
            assert E.augmented_window(a, rng).shape == (F.N_MELS, F.WINDOW_FRAMES)


def test_breakdown_counts_referred_rerecord_missed():
    import evaluation as E
    kids = [{"label": PRESENT, "pred": PRESENT, "grade": "I/VI"},
            {"label": PRESENT, "pred": ABSENT, "grade": "I/VI"},
            {"label": PRESENT, "pred": UNKNOWN, "grade": "III/VI"},
            {"label": ABSENT, "pred": PRESENT, "grade": "nan"}]
    out = E.breakdown(kids, "grade", ["I/VI", "II/VI", "III/VI"], PRESENT, UNKNOWN, ABSENT)
    assert out == [{"group": "I/VI", "n": 2, "referred": 1, "rerecord": 0, "missed": 1},
                   {"group": "III/VI", "n": 1, "referred": 0, "rerecord": 1, "missed": 0}]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_every_text_is_translated():
    out = subprocess.run(["node", str(ROOT / "tests" / "i18n_check.cjs")], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_wav_encoder_round_trips(tmp_path):
    script = tmp_path / "enc.cjs"
    script.write_text(
        "const fs=require('fs'),vm=require('vm');"
        "const ctx=vm.createContext({});"
        f"vm.runInContext(fs.readFileSync({json.dumps(str(ROOT / 'web' / 'js' / 'guided.js'))},'utf8')+';this.G=Guided;',ctx);"
        "const x=new Float32Array(48000);for(let i=0;i<x.length;i++)x[i]=0.5*Math.sin(2*Math.PI*220*i/48000);"
        f"fs.writeFileSync({json.dumps(str(tmp_path / 'out.wav'))},Buffer.from(ctx.G.encodeWav(x,48000)));")
    subprocess.run(["node", str(script)], check=True)
    from scipy.io import wavfile
    rate, y = wavfile.read(tmp_path / "out.wav")
    assert rate == 48000 and len(y) == 48000
    assert np.abs(y).max() == pytest.approx(0.5 * 32767, rel=0.01)
