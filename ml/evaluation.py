"""
Helpers for the extra evaluation that train.py runs inside cross-validation:

- degraded test audio (noise, phone-like microphone, handling rumble)
- training-time augmentation for the noise-robust model
- missed murmurs by grade / timing / age
- picking recordings for the "Train your ear" quiz
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

import features as F
from heartbeat_check import beat_measures

ROOT = Path(__file__).resolve().parent.parent
CHECK = json.loads((ROOT / "web" / "model" / "heartbeat_check.json").read_text())

# Test conditions. "noise*" and "phone" are the same kinds of damage the robust
# model sees in training; "rumble" (low-frequency handling noise) is not, so it
# shows whether robustness carries over to damage the model was never shown.
CONDITIONS = ["clean", "noise20", "noise10", "noise5", "phone", "rumble"]
CONDITION_NAMES = {
    "clean": "Original recordings",
    "noise20": "Background noise, 20 dB SNR",
    "noise10": "Background noise, 10 dB SNR",
    "noise5": "Background noise, 5 dB SNR",
    "phone": "Phone-like microphone (no sound below 150 Hz)",
    "rumble": "Handling rumble, 10 dB SNR (never seen in training)",
}
PHONE = butter(4, 150, "highpass", fs=F.SAMPLE_RATE, output="sos")
RUMBLE_SHAPE = butter(2, [5, 60], "bandpass", fs=F.SAMPLE_RATE, output="sos")


def add_noise(x, snr_db, rng, noise=None):
    noise = rng.normal(size=len(x)) if noise is None else noise
    scale = np.sqrt(np.mean(x ** 2) / (np.mean(noise ** 2) + 1e-12)) / 10 ** (snr_db / 20)
    return x + noise * scale


def degrade(x, condition, rng):
    if condition == "clean":
        return x
    if condition == "phone":
        return sosfilt(PHONE, x)
    if condition == "rumble":
        return add_noise(x, 10, rng, sosfilt(RUMBLE_SHAPE, rng.normal(size=len(x))))
    return add_noise(x, {"noise20": 20, "noise10": 10, "noise5": 5}[condition], rng)


WINDOW_SAMPLES = F.N_FFT + (F.WINDOW_FRAMES - 1) * F.HOP   # 4 seconds
MARGIN = 1000                                               # lets the filter settle


def augmented_window(audio, rng):
    """A random 4-second training window, damaged at random like a cheap
    microphone in a noisy room would: a missing low end (high-pass at a random
    60-250 Hz) and/or background noise (3-30 dB SNR). 40% stay clean."""
    need = WINDOW_SAMPLES + MARGIN
    if len(audio) < need:
        audio = np.tile(audio, int(np.ceil(need / len(audio))))
    start = int(rng.integers(0, len(audio) - need + 1))
    x = audio[start:start + need].astype(np.float64) / 32768.0
    if rng.random() < 0.6:
        if rng.random() < 0.5:
            x = sosfilt(butter(4, rng.uniform(60, 250), "highpass", fs=F.SAMPLE_RATE, output="sos"), x)
        if rng.random() < 0.6:
            x = add_noise(x, rng.uniform(3, 30), rng)
    return F.log_mel(x[MARGIN:])


def beat_ok(spec):
    rhythm, depth = beat_measures(spec)
    return bool(rhythm >= CHECK["rhythm_min"] and depth >= CHECK["depth_min"])


def breakdown(children, key, order, present, unknown, absent):
    """For children with a murmur: how many were referred, asked to re-record, or missed."""
    out = []
    for value in order:
        group = [c for c in children if c["label"] == present and c[key] == value]
        if group:
            out.append({"group": value, "n": len(group),
                        "referred": sum(c["pred"] == present for c in group),
                        "rerecord": sum(c["pred"] == unknown for c in group),
                        "missed": sum(c["pred"] == absent for c in group)})
    return out


def by_age(children, present, absent):
    out = []
    for age in ["Neonate", "Infant", "Child", "Adolescent"]:
        m = [c for c in children if c["age"] == age and c["label"] == present]
        n = [c for c in children if c["age"] == age and c["label"] == absent]
        if m or n:
            out.append({"group": age, "murmur": len(m), "no_murmur": len(n),
                        "sensitivity": sum(c["pred"] == present for c in m) / len(m) if m else None,
                        "specificity": sum(c["pred"] != present for c in n) / len(n) if n else None})
    return out


def pick_quiz(recordings, present, absent, rng):
    """9 murmur recordings (3 per grade) and 9 without, all with a clear heartbeat.
    Each carries the answer of the cross-validation model that never heard it."""
    usable = [r for r in recordings if r["beat_ok"]]
    picks = []
    for grade in ["III/VI", "II/VI", "I/VI"]:
        pool = [r for r in usable if r["label"] == present and r["grade"] == grade]
        picks += list(rng.choice(pool, 3, replace=False))
    pool = [r for r in usable if r["label"] == absent and r["grade"] in ("nan", "")]
    picks += list(rng.choice(pool, 9, replace=False))
    return [picks[i] for i in rng.permutation(len(picks))]
