"""
External validation on data LittleHeart has never seen: the PhysioNet/CinC
Challenge 2016 heart sound database (3,240 recordings from six research
groups in different countries, mostly adults, different stethoscopes).

Important: its labels are "normal" vs "abnormal heart" (confirmed by
cardiac diagnosis), not "murmur" vs "no murmur". Many abnormal hearts have
a murmur, but not all, so this tests whether the murmur score transfers to
new devices and populations, not the exact screening task.

Runs the exported ONNX model with the same steps as the browser.

Run:  python ml/download_data.py --external
      python ml/external_validation.py [model folder, default web/model]
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

import evaluation as E
import features as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "physionet2016"


def load_4k(path):
    rate, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768.0
    if rate != F.SAMPLE_RATE:
        x = resample_poly(x, F.SAMPLE_RATE, rate)
    return x


def recording_present_prob(session, spec):
    windows = np.stack([F.cut_window(spec, s) for s in F.window_starts(spec.shape[1])])[:, None]
    probs = session.run(None, {"logmel": windows.astype(np.float32)})[0]
    return float(probs.mean(axis=0)[1])


def summarize(rows, threshold):
    ok = [r for r in rows if r["beat_ok"]]
    y = np.array([r["abnormal"] for r in ok])
    p = np.array([r["present_prob"] for r in ok])
    flagged = p >= threshold
    out = {"recordings": len(rows), "abnormal_share": float(np.mean([r["abnormal"] for r in rows])),
           "beat_pass": len(ok) / len(rows)}
    if 0 < y.sum() < len(y):
        out |= {"auc": float(roc_auc_score(y, p)),
                "sensitivity": float(flagged[y == 1].mean()),
                "specificity": float((~flagged[y == 0]).mean())}
    return out


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "web" / "model"
    session = ort.InferenceSession(str(model_dir / "murmur_cnn.onnx"))
    threshold = json.loads((model_dir / "config.json").read_text())["thresholds"]["present"]

    rows = []
    for folder in sorted(DATA.glob("training-*")):
        for name, label in csv.reader(open(folder / "REFERENCE.csv")):
            spec = F.log_mel(load_4k(folder / f"{name}.wav"))
            rows.append({"subset": folder.name[-1], "abnormal": label.strip() == "1",
                         "beat_ok": E.beat_ok(spec), "present_prob": recording_present_prob(session, spec)})
        print(f"{folder.name}: {sum(r['subset'] == folder.name[-1] for r in rows)} recordings", flush=True)

    result = {
        "dataset": "PhysioNet/CinC Challenge 2016 training set (normal vs abnormal heart sounds)",
        "threshold": threshold,
        "overall": summarize(rows, threshold),
        "subsets": {s: summarize([r for r in rows if r["subset"] == s], threshold)
                    for s in sorted({r["subset"] for r in rows})},
    }
    (model_dir / "external.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
