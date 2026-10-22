"""
Builds the two PDFs required for the UnivaBio submission:

  docs/LittleHeart_Project_Description.pdf   one page
  docs/LittleHeart_Code.pdf                  all source code with line numbers

Run after training:  python docs/make_pdfs.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ACCENT = colors.HexColor("#c2364f")
GREY = colors.HexColor("#675f63")

CODE_FILES = [
    ("ml/download_data.py", "Downloads the CirCor and PhysioNet 2016 datasets from PhysioNet's open-data mirror."),
    ("ml/features.py", "Turns a recording into a log-mel spectrogram and 4-second windows."),
    ("ml/train.py", "Cross-validated training (clean or noise-robust), evaluation, ONNX export, demo and quiz sets."),
    ("ml/evaluation.py", "Damaged-audio test conditions, training augmentation, error analysis, quiz selection."),
    ("ml/heartbeat_check.py", "Safety check calibration: refuse recordings without a heartbeat rhythm."),
    ("ml/external_validation.py", "Tests the exported model on the PhysioNet 2016 dataset it never saw."),
    ("ml/compare_models.py", "Adds the clean-trained model's results for before/after comparison."),
    ("web/index.html", "Page structure: screening, ear training, screening log, validation and education tabs."),
    ("web/js/i18n.js", "Translations (English, Turkish, Portuguese, Spanish)."),
    ("web/js/features.js", "JavaScript port of features.py and the heartbeat check, so the browser computes identical numbers."),
    ("web/js/model.js", "Runs the ONNX model in the browser and applies the referral rule."),
    ("web/js/draw.js", "Spectrogram, murmur timeline, playback cursor and live waveform drawing."),
    ("web/js/app.js", "Screening view: real cases, uploads, verdict, per-valve results."),
    ("web/js/guided.js", "Guided recording with live heartbeat and sound-level check."),
    ("web/js/quiz.js", "Train-your-ear mode."),
    ("web/js/log.js", "Screening log stored on the device, with CSV export."),
    ("web/js/validation.js", "Validation tab: metrics, error analysis, robustness, external test."),
    ("web/js/main.js", "Start-up: tabs, language, modules, offline support."),
    ("web/sw.js", "Service worker that keeps the app working offline."),
    ("web/css/style.css", "Visual design, light/dark themes, phone and print layouts."),
    ("tests/test_pipeline.py", "Automated tests, including Python vs JavaScript parity and translations."),
    ("tests/js_logmel.cjs", "Node.js helper used by the parity test."),
    ("tests/i18n_check.cjs", "Checks that every text exists in all four languages."),
    ("docs/make_pdfs.py", "Generates this PDF and the one-page description."),
]


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=21, textColor=ACCENT, spaceAfter=1),
        "tagline": ParagraphStyle("tg", parent=base["Normal"], alignment=TA_CENTER, fontSize=9.5, textColor=GREY, spaceAfter=5),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontSize=10.2, textColor=ACCENT, spaceBefore=4, spaceAfter=1),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=8.3, leading=10.7),
        "small": ParagraphStyle("s", parent=base["Normal"], fontSize=7.3, leading=9, textColor=GREY),
        "code": ParagraphStyle("c", parent=base["Code"], fontSize=6.8, leading=8.3, leftIndent=0),
    }


def build_description(m, ext):
    s = styles()
    c, b, ds = m["cnn"], m["baseline"], m["dataset"]
    pct = lambda x: f"{round(x * 100)}%"
    rob = {r["condition"]: r for r in m["analysis"]["robustness"]}
    old = {r["condition"]: r for r in m["comparison"]["robustness"]} if "comparison" in m else None
    grades = {g["group"]: g for g in m["analysis"]["by_grade"]}
    missed = sum(g["missed"] for g in grades.values())
    faint = grades.get("I/VI", {"missed": 0})["missed"]
    loud = grades.get("III/VI")
    phone = (f"With a phone-like microphone, training on randomly damaged audio raised murmurs caught from "
             f"{pct(old['phone']['sensitivity'])} to <b>{pct(rob['phone']['sensitivity'])}</b>. ") if old else ""

    left = [
        Paragraph("The problem", s["h"]),
        Paragraph("About 1 in 100 babies is born with a heart defect, and rheumatic heart disease affects tens of millions of "
                  "people, most of them young and living in low- and middle-income countries. Both are often first noticed as a "
                  "heart murmur. But recognising a murmur takes years of training, and the echocardiogram that confirms it is "
                  "scarce exactly where these diseases are most common.", s["body"]),
        Paragraph("Our solution", s["h"]),
        Paragraph("<b>LittleHeart</b> lets a community health worker, school nurse or GP record a child's heart at the four "
                  "valve areas and instantly learn: <i>refer for echocardiogram</i>, <i>no murmur detected</i>, or <i>recording "
                  "unclear, record again</i>, so scarce echocardiograms go to the children who need them.", s["body"]),
        Paragraph("Key features", s["h"]),
        Paragraph("• <b>Guided recording</b>: step-by-step through the four sites with a live heartbeat and sound-level check.<br/>"
                  "• <b>Explainable results</b>: chest diagram, spectrogram per valve and a murmur-likelihood timeline.<br/>"
                  "• <b>Train your ear</b>: health workers practise on 18 real recordings against the AI.<br/>"
                  "• <b>Screening log</b> for school or village screening days, with CSV export.<br/>"
                  "• <b>Works offline</b>, installs like an app, and runs privately in the browser: no audio is uploaded.<br/>"
                  "• <b>4 languages</b> (English, Turkish, Portuguese, Spanish); real held-out cases; printable referral note.<br/>"
                  "• <b>Transparent validation tab</b> with an honest \"working / not yet / limits\" table.", s["body"]),
        Paragraph("Honest limits", s["h"]),
        Paragraph("A screening aid, not a diagnosis or medical device. It misses some (mostly faint) murmurs, was trained on one "
                  "population and one stethoscope, cannot tell innocent from pathological murmurs, and has not been tested with "
                  "real phone adapters or prospectively. All of this is stated in the app.", s["body"]),
    ]
    right = [
        Paragraph("How the AI works", s["h"]),
        Paragraph(f"Trained on the CirCor DigiScope dataset (PhysioNet): {ds['recordings']:,} recordings from {ds['children']} "
                  "Brazilian children, labelled by cardiologists. Each recording becomes a log-mel spectrogram (25-1000 Hz), cut "
                  "into 4-second windows that a small convolutional network (about 61,000 weights, PyTorch, run in the browser "
                  "with ONNX Runtime Web) scores as murmur / no murmur / unclear. During training, audio is randomly damaged with "
                  "noise and phone-like filtering. A <b>heartbeat safety check</b> refuses recordings with no 40-220 bpm rhythm, "
                  "so the AI never guesses on noise or speech.", s["body"]),
        Paragraph("Results", s["h"]),
        Paragraph(f"{m['folds']}-fold cross-validation, split by child, {ds['cv_children']} children: PhysioNet 2022 murmur score "
                  f"<b>{c['weighted_accuracy']['mean']:.2f} ± {c['weighted_accuracy']['std']:.2f}</b> (baseline "
                  f"{b['weighted_accuracy']['mean']:.2f}; the best challenge team scored 0.78 on a harder hidden test set, not a "
                  f"like-for-like comparison). It caught <b>{pct(c['sensitivity']['mean'])}</b> of children with murmurs and "
                  f"cleared <b>{pct(c['specificity']['mean'])}</b> of those without (AUC {c['auc']['mean']:.2f}). "
                  f"Of {missed} murmurs missed, {faint} were the faintest grade; loud grade III/VI murmurs were referred "
                  f"{pct(loud['referred'] / loud['n']) if loud else '-'} of the time. " + phone, s["body"]),
        Paragraph("External test", s["h"]),
        Paragraph(f"On the PhysioNet 2016 database (3,240 recordings, mostly adults, other devices, never used in training), "
                  f"the murmur score separated normal from abnormal hearts with AUC <b>{ext['overall']['auc']:.2f}</b>, and only "
                  f"{pct(ext['overall']['beat_pass'])} of recordings passed the heartbeat check. Its labels are not murmurs, but "
                  "the result shows LittleHeart must be validated locally before use in a new setting.", s["body"]),
        Paragraph("Next steps", s["h"]),
        Paragraph("Test with low-cost stethoscope adapters, validate with children in a new region, add innocent-vs-"
                  "pathological classification with echocardiography-labelled data, and pilot with a school RHD-screening "
                  "programme.", s["body"]),
    ]

    story = [Paragraph("LittleHeart", s["title"]),
             Paragraph("AI heart murmur screening for children · UnivaBio 2026: AI for Human Health", s["tagline"])]
    cols = Table([[left, right]], colWidths=[92 * mm, 92 * mm])
    cols.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (0, 0), 4 * mm)]))
    story += [cols, Spacer(1, 2.5 * mm)]
    shot = DOCS / "screenshot.png"
    if shot.exists():
        img = Image(str(shot))
        img._restrictSize(150 * mm, 72 * mm)
        story.append(img)
    story += [Spacer(1, 1.5 * mm),
              Paragraph("Tech: Python, PyTorch, scikit-learn, NumPy (training) · HTML, CSS, JavaScript, ONNX Runtime Web (app) · "
                        "Data: CirCor DigiScope (Oliveira et al. 2022) and PhysioNet/CinC Challenge 2016 (Liu et al. 2016), PhysioNet.",
                        s["small"])]

    doc = SimpleDocTemplate(str(DOCS / "LittleHeart_Project_Description.pdf"), pagesize=A4, leftMargin=13 * mm,
                            rightMargin=13 * mm, topMargin=10 * mm, bottomMargin=9 * mm, title="LittleHeart - Project Description")
    doc.build(story)
    return doc


def build_code_pdf():
    s = styles()
    story = [Paragraph("LittleHeart: source code", s["title"]),
             Paragraph("UnivaBio 2026 · AI for Human Health", s["tagline"]), Spacer(1, 4 * mm),
             Paragraph("Project structure", s["h"])]
    for path, purpose in CODE_FILES:
        story.append(Paragraph(f"<b>{path}</b>: {purpose}", s["body"]))
    story.append(Paragraph("Generated, not listed: web/model/murmur_cnn.onnx (trained network), config.json (spectrogram "
                           "settings and thresholds), metrics.json/.js (validation results), external.json (external test), "
                           "heartbeat_check.json, web/samples and web/quiz (recordings from the dataset). Third-party: "
                           "web/vendor/ort (ONNX Runtime Web, MIT licence).", s["body"]))
    for path, purpose in CODE_FILES:
        story += [PageBreak(), Paragraph(path, s["h"]), Paragraph(purpose, s["small"]), Spacer(1, 2 * mm)]
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
        story.append(Preformatted("\n".join(f"{i:4d}  {line}" for i, line in enumerate(lines, 1)),
                                  s["code"], maxLineLength=120, newLineChars="      "))

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 13 * mm, 8 * mm, f"LittleHeart · page {doc.page}")

    doc = SimpleDocTemplate(str(DOCS / "LittleHeart_Code.pdf"), pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm,
                            topMargin=12 * mm, bottomMargin=14 * mm, title="LittleHeart - Source Code")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return doc


if __name__ == "__main__":
    metrics = json.loads((ROOT / "web" / "model" / "metrics.json").read_text())
    external = json.loads((ROOT / "web" / "model" / "external.json").read_text())
    d = build_description(metrics, external)
    c = build_code_pdf()
    print(f"Description: {d.page} page(s); code listing: {c.page} page(s)")
