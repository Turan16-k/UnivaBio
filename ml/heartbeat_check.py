"""
Safety check: does a recording contain a heartbeat rhythm at all?

The CNN was only trained on heart sounds, so it can give confident but
meaningless answers for speech, noise or a phone lying on a table. Before
trusting the CNN, we look for a repeating beat between 40 and 220 beats per
minute in the loudness of the low frequencies (where S1 and S2 live).

The threshold is calibrated so that 98% of real CirCor recordings pass,
then tested on signals that are not heart sounds.

Run:  python ml/heartbeat_check.py      (after ml/train.py has cached spectrograms)
"""

import json
from pathlib import Path

import numpy as np

import features as F

ROOT = Path(__file__).resolve().parent.parent
FRAME_RATE = F.SAMPLE_RATE / F.HOP           # 62.5 spectrogram frames per second
LOW_BANDS = 12                                # mel bands up to roughly 200 Hz
MIN_LAG = int(FRAME_RATE * 60 / 220)          # 220 bpm
MAX_LAG = int(FRAME_RATE * 60 / 40)           # 40 bpm


def beat_measures(spec):
    """Returns (rhythm, depth) of the low-frequency loudness curve.
    rhythm: highest autocorrelation at a heartbeat-like delay (0 = none, 1 = perfectly regular).
    depth:  how much the loudness rises and falls (standard deviation, in log10 units);
            heartbeats are loud clicks with quiet gaps, a steady tone is flat."""
    env = spec[:LOW_BANDS].mean(axis=0).astype(np.float64)
    env = env - env.mean()
    denom = np.dot(env, env)
    depth = float(np.sqrt(denom / max(1, len(env))))
    if denom <= 0 or len(env) <= MAX_LAG + 1:
        return 0.0, depth
    best = 0.0
    for lag in range(MIN_LAG, MAX_LAG + 1):
        best = max(best, np.dot(env[:-lag], env[lag:]) / denom)
    return float(best), depth


def synthetic_negatives(rng, seconds=15):
    t = np.arange(F.SAMPLE_RATE * seconds) / F.SAMPLE_RATE
    n = len(t)
    syllables = np.zeros(n)
    pos = 0
    while pos < n:  # irregular syllables (0.08-0.35 s) separated by irregular gaps
        length = int(rng.uniform(0.08, 0.35) * F.SAMPLE_RATE)
        syllables[pos:pos + length] = rng.uniform(0.5, 1.0)
        pos += length + int(rng.uniform(0.03, 0.4) * F.SAMPLE_RATE)
    pitch = 120 + 30 * np.sin(2 * np.pi * 0.3 * t)
    return {
        "white noise": rng.normal(size=n),
        "pink-ish noise": np.cumsum(rng.normal(size=n)) * 0.01 + rng.normal(size=n),
        "steady tone 440 Hz": np.sin(2 * np.pi * 440 * t),
        "mains hum 50 Hz": np.sin(2 * np.pi * 50 * t) + 0.1 * rng.normal(size=n),
        "speech-like babble": syllables * np.sin(2 * np.pi * np.cumsum(pitch) / F.SAMPLE_RATE) + 0.05 * rng.normal(size=n),
        "silence with hiss": 0.001 * rng.normal(size=n),
    }


def main():
    specs = np.load(ROOT / "data" / "spectrograms.npz", allow_pickle=True)["specs"].item()
    real = np.array([beat_measures(s) for s in specs.values()])
    # The rhythm limit lets 97% of real recordings through, the depth limit 99%.
    t_rhythm = float(np.round(np.percentile(real[:, 0], 3), 3))
    t_depth = float(np.round(np.percentile(real[:, 1], 1), 3))
    passes = lambda m: m[0] >= t_rhythm and m[1] >= t_depth
    real_pass = float(np.mean([passes(m) for m in real]))
    print(f"{len(real)} real recordings: median rhythm {np.median(real[:, 0]):.3f}, depth {np.median(real[:, 1]):.3f}")
    print(f"limits: rhythm >= {t_rhythm}, depth >= {t_depth}; real recordings passing: {real_pass:.1%}")

    negatives = {}
    trials = [synthetic_negatives(np.random.default_rng(seed)) for seed in range(10)]
    for name in trials[0]:
        rejected = sum(not passes(beat_measures(F.log_mel(t[name]))) for t in trials)
        negatives[name] = {"rejected": rejected, "trials": len(trials)}
        print(f"  {name:20s} rejected {rejected}/{len(trials)}")

    out = {
        "rhythm_min": t_rhythm,
        "depth_min": t_depth,
        "low_bands": LOW_BANDS,
        "min_lag": MIN_LAG,
        "max_lag": MAX_LAG,
        "real_pass_rate": real_pass,
        "real_recordings": int(len(real)),
        "negatives": negatives,
    }
    (ROOT / "web" / "model" / "heartbeat_check.json").write_text(json.dumps(out, indent=1))
    print("Saved web/model/heartbeat_check.json")


if __name__ == "__main__":
    main()
