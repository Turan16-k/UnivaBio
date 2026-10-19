"""
LittleHeart - training and evaluation.

1. Reads the CirCor DigiScope dataset (942 children, 3,163 heart sound recordings).
2. Holds out a few children as demo cases that the model never trains on.
3. Runs 5-fold cross-validation, split by CHILD (never by recording), comparing
   our CNN against a simple baseline using the official PhysioNet 2022 score.
4. Trains the final model on all remaining children and exports it to ONNX
   for the browser, plus the metrics and demo cases.

Run:  python ml/train.py            (about 30-60 minutes on a laptop CPU)
      python ml/train.py --quick    (2 folds, few epochs, to test the pipeline)
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import features as F

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = next((ROOT / "data").glob("*circor*1.0.3*"), ROOT / "data" / "circor")
CACHE = ROOT / "data" / "spectrograms.npz"
WEB_MODEL = ROOT / "web" / "model"
WEB_SAMPLES = ROOT / "web" / "samples"

CLASSES = ["Absent", "Present", "Unknown"]
ABSENT, PRESENT, UNKNOWN = 0, 1, 2
SEED = 7


# ---------------------------------------------------------------- data

def load_patients():
    rows = list(csv.DictReader(open(DATA_DIR / "training_data.csv", encoding="utf-8")))
    patients = []
    for r in rows:
        pid = r["Patient ID"]
        murmur_locs = set(r["Murmur locations"].split("+")) if r["Murmur locations"] not in ("", "nan") else set()
        recs = []
        for wav in sorted((DATA_DIR / "training_data").glob(f"{pid}_*.wav")):
            loc = wav.stem.split("_")[1]
            if r["Murmur"] == "Present":
                # A murmur is only audible at some valves; other sites are labelled absent.
                label = PRESENT if loc in murmur_locs else ABSENT
            elif r["Murmur"] == "Unknown":
                label = UNKNOWN
            else:
                label = ABSENT
            recs.append({"path": wav, "location": loc, "label": label})
        patients.append({
            "id": pid,
            "label": CLASSES.index(r["Murmur"]),
            "age": r["Age"],
            "sex": r["Sex"],
            "murmur_locations": sorted(murmur_locs),
            "timing": r["Systolic murmur timing"],
            "grade": r["Systolic murmur grading"],
            "outcome": r["Outcome"],
            "recordings": recs,
        })
    return patients


def load_spectrograms(patients):
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=True)
        specs = cached["specs"].item()
    else:
        specs = {}
    missing = [rec for p in patients for rec in p["recordings"] if rec["path"].name not in specs]
    if missing:
        print(f"Computing {len(missing)} spectrograms...")
        for rec in missing:
            specs[rec["path"].name] = F.log_mel(F.load_wav(rec["path"]))
        np.savez_compressed(CACHE, specs=np.array(specs, dtype=object))
    for p in patients:
        for rec in p["recordings"]:
            rec["spec"] = specs[rec["path"].name]


# ---------------------------------------------------------------- model

class MurmurCNN(nn.Module):
    """Small convolutional network: four conv blocks, then average over time
    and frequency, then 3 outputs (Absent / Present / Unknown)."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 1, -1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 1, -1, 1))

        def block(c_in, c_out, pool=True):
            layers = [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers

        self.body = nn.Sequential(*block(1, 16), *block(16, 32), *block(32, 64), *block(64, 64, pool=False))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 3))

    def forward(self, x):                  # x: (batch, 1, N_MELS, WINDOW_FRAMES) log-mel
        x = (x - self.mean) / self.std     # standardise each mel band
        x = self.body(x)
        x = x.mean(dim=(2, 3))             # global average pooling
        return self.head(x)


class WithSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def band_stats(recordings):
    allspec = np.concatenate([r["spec"] for r in recordings], axis=1)
    return allspec.mean(axis=1), allspec.std(axis=1) + 1e-6


def random_window(spec, rng):
    n = spec.shape[1]
    start = 0 if n <= F.WINDOW_FRAMES else int(rng.integers(0, n - F.WINDOW_FRAMES + 1))
    return F.cut_window(spec, start)


def train_cnn(train_recs, val_patients, epochs, rng, log=print):
    torch.manual_seed(int(rng.integers(1e9)))
    mean, std = band_stats(train_recs)
    model = MurmurCNN(mean, std)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # Murmurs are rarer and matter more, so their mistakes cost more.
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 2.0]))
    labels = np.array([r["label"] for r in train_recs])

    best = (-1.0, None, None)
    for epoch in range(epochs):
        model.train()
        order = np.concatenate([rng.permutation(len(train_recs)) for _ in range(3)])
        total = 0.0
        for i in range(0, len(order), 64):
            batch = order[i:i + 64]
            x = np.stack([random_window(train_recs[j]["spec"], rng) for j in batch])[:, None]
            x = x + rng.normal(0, 0.05, x.shape).astype(np.float32)
            y = torch.tensor(labels[batch])
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(x)), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        sched.step()

        probs = patient_probs(lambda w: predict_windows(model, w), val_patients)
        t, score = tune_thresholds(probs, [p["label"] for p in val_patients])
        log(f"    epoch {epoch + 1:2d}  loss {total / len(order):.3f}  val weighted acc {score:.3f}")
        if score > best[0]:
            best = (score, {k: v.clone() for k, v in model.state_dict().items()}, t)
    model.load_state_dict(best[1])
    model.eval()
    return model, best[2]


@torch.no_grad()
def predict_windows(model, windows):
    model.eval()
    x = torch.from_numpy(np.stack(windows)[:, None].astype(np.float32))
    return torch.softmax(model(x), dim=1).numpy()


def recording_probs(predict, spec):
    """Average the class probabilities of every 4-second window."""
    windows = [F.cut_window(spec, s) for s in F.window_starts(spec.shape[1])]
    return predict(windows).mean(axis=0)


def patient_probs(predict, patients):
    """For each child: the highest Present and Unknown probability over all valves."""
    out = []
    for p in patients:
        rec = np.array([recording_probs(predict, r["spec"]) for r in p["recordings"]])
        out.append((rec[:, PRESENT].max(), rec[:, UNKNOWN].max()))
    return np.array(out)


# ---------------------------------------------------------------- decisions and metrics

def decide(probs, t_present, t_unknown):
    """Present if any valve looks like a murmur, else Unknown if any recording
    is unclear, else Absent. Same rule order as the PhysioNet challenge."""
    out = np.full(len(probs), ABSENT)
    out[probs[:, 1] >= t_unknown] = UNKNOWN
    out[probs[:, 0] >= t_present] = PRESENT
    return out


def weighted_accuracy(y_true, y_pred):
    """Official PhysioNet 2022 murmur score: Present counts 5x, Unknown 3x, Absent 1x."""
    w = {PRESENT: 5, UNKNOWN: 3, ABSENT: 1}
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    num = sum(w[c] * np.sum((y_true == c) & (y_pred == c)) for c in w)
    den = sum(w[c] * np.sum(y_true == c) for c in w)
    return float(num / den)


def tune_thresholds(probs, y_true):
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best = (-1.0, (0.5, 0.5))
    for tp in grid:
        for tu in grid:
            s = weighted_accuracy(y_true, decide(probs, tp, tu))
            if s > best[0]:
                best = (s, (float(tp), float(tu)))
    return best[1], best[0]


def screening_metrics(y_true, y_pred, present_prob):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    known = y_true != UNKNOWN
    is_murmur = y_true[known] == PRESENT
    flagged = y_pred[known] == PRESENT
    confusion = [[int(np.sum((y_true == t) & (y_pred == p))) for p in range(3)] for t in range(3)]
    return {
        "weighted_accuracy": weighted_accuracy(y_true, y_pred),
        "sensitivity": float(np.mean(flagged[is_murmur])),
        "specificity": float(np.mean(~flagged[~is_murmur])),
        "auc": float(roc_auc_score(is_murmur, present_prob[known])),
        "confusion": confusion,  # rows = true, cols = predicted (Absent, Present, Unknown)
    }


# ---------------------------------------------------------------- baseline

def baseline_features(spec):
    return np.concatenate([spec.mean(axis=1), spec.std(axis=1)])


def train_baseline(train_recs):
    X = np.stack([baseline_features(r["spec"]) for r in train_recs])
    y = np.array([r["label"] for r in train_recs])
    clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf.fit((X - mean) / std, y)
    return lambda spec: clf.predict_proba(((baseline_features(spec) - mean) / std)[None])[0]


def baseline_patient_probs(predict_rec, patients):
    out = []
    for p in patients:
        rec = np.array([predict_rec(r["spec"]) for r in p["recordings"]])
        out.append((rec[:, PRESENT].max(), rec[:, UNKNOWN].max()))
    return np.array(out)


# ---------------------------------------------------------------- main

def pick_demo_patients(patients, rng):
    """A few children per class that are never used for training or CV."""
    chosen = []
    for label, n in [(PRESENT, 4), (ABSENT, 4), (UNKNOWN, 2)]:
        pool = [p for p in patients if p["label"] == label and 3 <= len(p["recordings"]) <= 5]
        chosen += list(rng.choice(pool, n, replace=False))
    return chosen


def summarize(fold_metrics):
    keys = ["weighted_accuracy", "sensitivity", "specificity", "auc"]
    return {k: {"mean": float(np.mean([m[k] for m in fold_metrics])),
                "std": float(np.std([m[k] for m in fold_metrics]))} for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_folds, epochs = (2, 2) if args.quick else (5, 20)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    patients = load_patients()
    load_spectrograms(patients)
    demo = pick_demo_patients(patients, rng)
    demo_ids = {p["id"] for p in demo}
    pool = [p for p in patients if p["id"] not in demo_ids]
    n_recs = sum(len(p["recordings"]) for p in patients)
    print(f"{len(patients)} children, {n_recs} recordings; {len(demo)} held out as demo cases")

    y_pool = np.array([p["label"] for p in pool])
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=SEED).split(np.zeros(len(pool)), y_pool)
    cnn_folds, base_folds = [], []
    oof_true, oof_pred = [], []

    for k, (tr_idx, te_idx) in enumerate(folds):
        train_p = [pool[i] for i in tr_idx]
        test_p = [pool[i] for i in te_idx]
        inner_train, inner_val = train_test_split(
            train_p, test_size=0.15, random_state=SEED, stratify=[p["label"] for p in train_p])
        train_recs = [r for p in inner_train for r in p["recordings"]]
        y_test = [p["label"] for p in test_p]
        print(f"Fold {k + 1}/{n_folds}: train {len(inner_train)}, val {len(inner_val)}, test {len(test_p)} children")

        model, (tp, tu) = train_cnn(train_recs, inner_val, epochs, rng)
        probs = patient_probs(lambda w: predict_windows(model, w), test_p)
        pred = decide(probs, tp, tu)
        m = screening_metrics(y_test, pred, probs[:, 0])
        cnn_folds.append(m)
        oof_true += y_test
        oof_pred += list(pred)

        base = train_baseline(train_recs)
        bt, _ = tune_thresholds(baseline_patient_probs(base, inner_val), [p["label"] for p in inner_val])
        bprobs = baseline_patient_probs(base, test_p)
        base_folds.append(screening_metrics(y_test, decide(bprobs, *bt), bprobs[:, 0]))
        print(f"  CNN weighted acc {m['weighted_accuracy']:.3f} (sens {m['sensitivity']:.2f}, "
              f"spec {m['specificity']:.2f}, AUC {m['auc']:.3f})  |  baseline {base_folds[-1]['weighted_accuracy']:.3f}")

    cnn_summary, base_summary = summarize(cnn_folds), summarize(base_folds)
    confusion = [[int(np.sum((np.array(oof_true) == t) & (np.array(oof_pred) == p))) for p in range(3)] for t in range(3)]
    print("CNN     ", {k: round(v["mean"], 3) for k, v in cnn_summary.items()})
    print("Baseline", {k: round(v["mean"], 3) for k, v in base_summary.items()})

    print("Training final model on all non-demo children...")
    final_train, final_val = train_test_split(pool, test_size=0.15, random_state=SEED, stratify=y_pool)
    model, (tp, tu) = train_cnn([r for p in final_train for r in p["recordings"]], final_val, epochs, rng)
    export(model, tp, tu, cnn_summary, base_summary, confusion, n_folds, len(pool), demo, len(patients), n_recs)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


def export(model, tp, tu, cnn_summary, base_summary, confusion, n_folds, n_pool, demo, n_patients, n_recs):
    WEB_MODEL.mkdir(parents=True, exist_ok=True)
    WEB_SAMPLES.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, F.N_MELS, F.WINDOW_FRAMES)
    torch.onnx.export(WithSoftmax(model).eval(), dummy, str(WEB_MODEL / "murmur_cnn.onnx"),
                      input_names=["logmel"], output_names=["probs"],
                      dynamic_axes={"logmel": {0: "batch"}, "probs": {0: "batch"}},
                      opset_version=17, dynamo=False)

    config = {
        "features": {
            "sample_rate": F.SAMPLE_RATE, "n_fft": F.N_FFT, "hop": F.HOP, "n_mels": F.N_MELS,
            "window_frames": F.WINDOW_FRAMES, "window_step": F.WINDOW_STEP, "eps": F.EPS,
            "mel_filterbank": np.round(F.MEL_FB, 7).tolist(),
        },
        "classes": CLASSES,
        "thresholds": {"present": tp, "unknown": tu},
    }
    (WEB_MODEL / "config.json").write_text(json.dumps(config))

    demo_cases = []
    for p in demo:
        files = []
        for r in p["recordings"]:
            shutil.copy(r["path"], WEB_SAMPLES / r["path"].name)
            files.append({"file": r["path"].name, "location": r["location"]})
        demo_cases.append({"id": p["id"], "truth": CLASSES[p["label"]], "age": p["age"], "sex": p["sex"],
                           "murmur_locations": p["murmur_locations"], "timing": p["timing"],
                           "grade": p["grade"], "recordings": files})

    metrics = {
        "dataset": {"children": n_patients, "recordings": n_recs, "cv_children": n_pool, "demo_children": len(demo)},
        "folds": n_folds,
        "cnn": cnn_summary,
        "baseline": base_summary,
        "confusion": confusion,
        "challenge_reference": {"top_hidden_test": 0.780, "note": "Best team, PhysioNet 2022 hidden test set"},
        "demo_cases": demo_cases,
    }
    (WEB_MODEL / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (WEB_MODEL / "metrics.js").write_text("const METRICS = " + json.dumps(metrics) + ";\n")
    print(f"Exported model, thresholds ({tp}, {tu}) and {len(demo_cases)} demo cases to web/")


if __name__ == "__main__":
    main()
