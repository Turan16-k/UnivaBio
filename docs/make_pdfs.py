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
    ("ml/download_data.py", "Downloads the CirCor DigiScope dataset from PhysioNet's open-data mirror."),
    ("ml/features.py", "Turns a recording into a log-mel spectrogram and 4-second windows."),
    ("ml/train.py", "Cross-validated training of the CNN and baseline, ONNX export, demo cases."),
    ("ml/heartbeat_check.py", "Safety check calibration: refuse recordings without a heartbeat rhythm."),
    ("web/index.html", "Page structure: screening, validation and education tabs."),
    ("web/js/features.js", "JavaScript port of features.py, so the browser computes identical spectrograms."),
    ("web/js/model.js", "Runs the ONNX model in the browser and applies the referral rule."),
    ("web/js/app.js", "User interface: cases, uploads, microphone, spectrograms, results, validation charts."),
    ("web/css/style.css", "Visual design, light/dark themes and phone layout."),
    ("tests/test_pipeline.py", "Automated tests, including Python vs JavaScript feature parity."),
    ("tests/js_logmel.cjs", "Node.js helper used by the parity test."),
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


def build_description(m):
    s = styles()
    c, b, ds = m["cnn"], m["baseline"], m["dataset"]
    pct = lambda x: f"{round(x * 100)}%"

    left = [
        Paragraph("The problem", s["h"]),
        Paragraph("About 1 in 100 babies is born with a heart defect, and rheumatic heart disease affects tens of millions of "
                  "people, most of them young and living in low- and middle-income countries. Both are often first noticed as a heart murmur. "
                  "But recognising a murmur through a stethoscope takes years of training, and the echocardiogram that confirms "
                  "it, and the cardiologists who read it, are scarce exactly where these diseases are most common. Many children "
                  "are diagnosed only when the heart is already damaged.", s["body"]),
        Paragraph("Our solution", s["h"]),
        Paragraph("<b>LittleHeart</b> is a web app that lets a community health worker, school nurse or GP record a child's "
                  "heart at the four standard valve areas and instantly learn whether a murmur is likely: <i>refer for "
                  "echocardiogram</i>, <i>no murmur detected</i>, or <i>recording unclear, record again</i>. It turns an "
                  "expert skill into a consistent second opinion so scarce echocardiograms go to the children who need them.", s["body"]),
        Paragraph("Key features", s["h"]),
        Paragraph("• <b>Three ways in</b>: real held-out patient cases, file upload, or live microphone/stethoscope recording.<br/>"
                  "• <b>Per-valve results</b> on an interactive chest diagram.<br/>"
                  "• <b>Explainable</b>: spectrogram of each recording with a moving playhead and a murmur-likelihood timeline.<br/>"
                  "• <b>Private and offline-capable</b>: the AI runs in the browser (ONNX Runtime Web); no audio is uploaded.<br/>"
                  "• <b>Transparent validation tab</b> and an honest \"working / not yet / limits\" table.<br/>"
                  "• Education tab on murmurs, recording technique and next steps; phone layout and dark mode.", s["body"]),
    ]
    right = [
        Paragraph("How the AI works", s["h"]),
        Paragraph(f"Trained on the CirCor DigiScope dataset (PhysioNet): {ds['recordings']:,} recordings from {ds['children']} "
                  "children in Brazil, labelled by cardiologists. Each recording becomes a log-mel spectrogram (25-1000 Hz), "
                  "cut into 4-second windows. A small convolutional neural network (about 61,000 weights, PyTorch) scores each "
                  "window as murmur / no murmur / unclear; scores are averaged per valve, and a child is referred if any valve "
                  "crosses a tuned threshold. A <b>heartbeat safety check</b> first refuses recordings with no 40-220 bpm "
                  "rhythm (noise, tones, most speech), so the AI never guesses on sounds it was not trained for. The same "
                  "code exists in Python and JavaScript and automated tests check they match.", s["body"]),
        Paragraph("Results", s["h"]),
        Paragraph(f"{m['folds']}-fold cross-validation, split by child, on {ds['cv_children']} children: official PhysioNet 2022 "
                  f"murmur score <b>{c['weighted_accuracy']['mean']:.2f} ± {c['weighted_accuracy']['std']:.2f}</b> "
                  f"(simple baseline {b['weighted_accuracy']['mean']:.2f}; best challenge team 0.78 on its hidden test set). "
                  f"It caught <b>{pct(c['sensitivity']['mean'])}</b> of children with murmurs and cleared "
                  f"<b>{pct(c['specificity']['mean'])}</b> of those without (AUC {c['auc']['mean']:.2f}). "
                  f"{ds['demo_children']} demo children were never used in training.", s["body"]),
        Paragraph("Honest limits", s["h"]),
        Paragraph("A screening aid, not a diagnosis or medical device. It still misses some murmurs, was trained on one "
                  "population and one stethoscope, cannot yet tell innocent from pathological murmurs, and has not been "
                  "tested prospectively or with phone microphones. All of this is stated in the app.", s["body"]),
        Paragraph("Next steps", s["h"]),
        Paragraph("Test with low-cost stethoscope adapters, validate on other datasets and populations, add murmur timing and "
                  "grading, multilingual interface, and a pilot with a school RHD-screening programme.", s["body"]),
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
        img._restrictSize(165 * mm, 98 * mm)
        story.append(img)
    story += [Spacer(1, 1.5 * mm),
              Paragraph("Tech: Python, PyTorch, scikit-learn, NumPy (training) · HTML, CSS, JavaScript, ONNX Runtime Web (app) · "
                        "Data: CirCor DigiScope Phonocardiogram Dataset, Oliveira et al. 2022, PhysioNet, CC BY 4.0.", s["small"])]

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
    story.append(Paragraph("Generated by ml/train.py (not listed): web/model/murmur_cnn.onnx (trained network), "
                           "config.json (spectrogram settings and thresholds), metrics.json/.js (validation results), "
                           "web/samples/*.wav (demo recordings from the dataset).", s["body"]))
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
    d = build_description(metrics)
    c = build_code_pdf()
    print(f"Description: {d.page} page(s); code listing: {c.page} page(s)")
