"""
Adds the clean-trained model's results to the deployed (robust) model's
metrics, so the Validation tab can show before -> after for damaged audio
and for the external dataset.

Inputs:
  clean metrics.json   from: python ml/train.py --no-final --out data/eval_clean
  clean external.json  from: python ml/external_validation.py <folder with the clean-trained ONNX model>
Run after the deployed model was trained (python ml/train.py --robust) and
externally tested (python ml/external_validation.py).

Usage: python ml/compare_models.py data/eval_clean/model/metrics.json data/external_clean_model.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOYED = ROOT / "web" / "model"


def main():
    clean = json.loads(Path(sys.argv[1]).read_text())
    clean_external = json.loads(Path(sys.argv[2]).read_text())
    metrics = json.loads((DEPLOYED / "metrics.json").read_text())
    metrics["comparison"] = {
        "note": "Same cross-validation folds, model trained on clean audio only",
        "cnn": clean["cnn"],
        "robustness": clean["analysis"]["robustness"],
        "external": {"overall": clean_external["overall"], "subsets": clean_external["subsets"]},
    }
    (DEPLOYED / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (DEPLOYED / "metrics.js").write_text("const METRICS = " + json.dumps(metrics) + ";\n")
    for c, r in zip(clean["analysis"]["robustness"], metrics["analysis"]["robustness"]):
        print(f"{r['condition']:8s} score {c['weighted_accuracy']:.3f} -> {r['weighted_accuracy']:.3f}   "
              f"sens {c['sensitivity']:.2f} -> {r['sensitivity']:.2f}   spec {c['specificity']:.2f} -> {r['specificity']:.2f}")


if __name__ == "__main__":
    main()
