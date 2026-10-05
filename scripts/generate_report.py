"""
scripts/generate_report.py
--------------------------
Generates a styled PDF evaluation report for the Physics RAG Chatbot.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --input logs/evaluation_results.json --output logs/report.pdf
"""

import argparse
import json
import io
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import BalancedColumns
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.pdfgen import canvas

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY       = colors.HexColor("#0d1b2a")
BLUE       = colors.HexColor("#1e3a5f")
ACCENT     = colors.HexColor("#4a9eff")
TEAL       = colors.HexColor("#00b4d8")
GREEN      = colors.HexColor("#22c55e")
AMBER      = colors.HexColor("#f59e0b")
RED        = colors.HexColor("#ef4444")
LIGHT_GRAY = colors.HexColor("#f1f5f9")
MID_GRAY   = colors.HexColor("#94a3b8")
WHITE      = colors.white
TEXT       = colors.HexColor("#1e293b")


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _save_fig(fig, dpi=150) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def make_gauge_chart(value: float, target: float, label: str,
                     above_is_good: bool = True) -> io.BytesIO:
    """Semi-circle gauge chart."""
    fig, ax = plt.subplots(figsize=(3.2, 2.0), subplot_kw=dict(aspect="equal"),
                           facecolor="#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    passed = (value >= target) if above_is_good else (value <= target)
    color  = "#22c55e" if passed else "#ef4444"
    bg     = "#1e3a5f"

    theta1, theta2 = 0, 180
    # Background arc
    arc_bg = mpatches.Wedge((0.5, 0), 0.42, theta1, theta2,
                             width=0.10, facecolor=bg, edgecolor="none")
    ax.add_patch(arc_bg)
    # Value arc
    val_angle = theta1 + (theta2 - theta1) * min(value, 1.0)
    arc_val = mpatches.Wedge((0.5, 0), 0.42, theta1, val_angle,
                              width=0.10, facecolor=color, edgecolor="none")
    ax.add_patch(arc_val)

    ax.text(0.5, 0.10, f"{value:.1%}", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, -0.05, label, ha="center", va="center",
            fontsize=7, color="#94a3b8", transform=ax.transAxes)
    status = "PASS" if passed else "FAIL"
    status_color = "#22c55e" if passed else "#ef4444"
    ax.text(0.5, -0.22, status, ha="center", va="center",
            fontsize=8, fontweight="bold", color=status_color,
            transform=ax.transAxes)

    ax.set_xlim(0, 1); ax.set_ylim(-0.3, 0.6)
    ax.axis("off")
    return _save_fig(fig)


def make_category_bar(per_cat: dict) -> io.BytesIO:
    """Horizontal bar chart of citation accuracy by category."""
    cats = [c for c in per_cat if c != "out_of_scope"]
    accs = [per_cat[c]["citation_accuracy"] for c in cats]
    confs = [per_cat[c]["mean_confidence"] for c in cats]

    y = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(6.5, 2.6), facecolor="#0d1b2a")
    ax.set_facecolor("#111827")

    bar_h = 0.35
    bars1 = ax.barh(y + bar_h/2, accs,  bar_h, color="#4a9eff", label="Citation Acc")
    bars2 = ax.barh(y - bar_h/2, confs, bar_h, color="#00b4d8", label="Confidence")

    ax.axvline(0.85, color="#f59e0b", linewidth=1.2, linestyle="--",
               label="85% target", alpha=0.8)

    for bar in bars1:
        w = bar.get_width()
        ax.text(min(w + 0.01, 1.02), bar.get_y() + bar.get_height()/2,
                f"{w:.0%}", va="center", fontsize=7, color="white")
    for bar in bars2:
        w = bar.get_width()
        ax.text(min(w + 0.01, 1.02), bar.get_y() + bar.get_height()/2,
                f"{w:.0%}", va="center", fontsize=7, color="#94a3b8")

    ax.set_yticks(y)
    ax.set_yticklabels([c.title() for c in cats], color="white", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.3,
              labelcolor="white", facecolor="#1e3a5f")
    ax.grid(axis="x", color="#1e3a5f", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return _save_fig(fig)


def make_confidence_scatter(results: list) -> io.BytesIO:
    """Scatter: question index vs confidence, coloured by category."""
    cat_colors = {
        "mechanics":       "#4a9eff",
        "electromagnetism":"#00b4d8",
        "quantum":         "#a78bfa",
        "thermodynamics":  "#f59e0b",
        "optics":          "#34d399",
        "out_of_scope":    "#6b7280",
    }
    fig, ax = plt.subplots(figsize=(6.5, 2.6), facecolor="#0d1b2a")
    ax.set_facecolor("#111827")

    for i, r in enumerate(results):
        cat = r.get("category", "mechanics")
        conf = r.get("confidence", 0)
        c = cat_colors.get(cat, "#ffffff")
        ax.scatter(i + 1, conf, color=c, s=60, zorder=3,
                   edgecolors="white", linewidths=0.4)

    ax.axhline(0.85, color="#f59e0b", linewidth=1.2, linestyle="--",
               alpha=0.8, label="85% target")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0, len(results) + 1)
    ax.set_xlabel("Question #", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Confidence", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    ax.grid(color="#1e3a5f", linewidth=0.5, alpha=0.5)

    legend_patches = [mpatches.Patch(color=v, label=k.title())
                      for k, v in cat_colors.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=6,
              framealpha=0.3, labelcolor="white", facecolor="#1e3a5f",
              ncol=2)
    fig.tight_layout()
    return _save_fig(fig)


# ── PDF page decorators ───────────────────────────────────────────────────────

class ReportCanvas:
    """Header/footer on every page."""

    def __init__(self, title: str, date: str):
        self.title = title
        self.date  = date

    def on_page(self, canv: canvas.Canvas, doc):
        w, h = A4
        # Top header bar
        canv.setFillColor(NAVY)
        canv.rect(0, h - 28*mm, w, 28*mm, fill=1, stroke=0)
        canv.setFillColor(ACCENT)
        canv.rect(0, h - 30*mm, w, 2*mm, fill=1, stroke=0)

        canv.setFillColor(WHITE)
        canv.setFont("Helvetica-Bold", 11)
        canv.drawString(1.5*cm, h - 16*mm, self.title)
        canv.setFont("Helvetica", 8)
        canv.setFillColor(MID_GRAY)
        canv.drawRightString(w - 1.5*cm, h - 16*mm, self.date)

        # Footer
        canv.setFillColor(NAVY)
        canv.rect(0, 0, w, 14*mm, fill=1, stroke=0)
        canv.setFillColor(ACCENT)
        canv.rect(0, 14*mm, w, 0.5*mm, fill=1, stroke=0)
        canv.setFillColor(MID_GRAY)
        canv.setFont("Helvetica", 7)
        canv.drawString(1.5*cm, 5*mm, "Physics RAG Chatbot — Confidential Evaluation Report")
        canv.drawRightString(w - 1.5*cm, 5*mm, f"Page {doc.page}")


# ── Styles ────────────────────────────────────────────────────────────────────

def get_styles():
    base = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle("h1", parent=base["Normal"],
                             fontSize=22, textColor=WHITE, fontName="Helvetica-Bold",
                             spaceAfter=6, alignment=TA_CENTER),
        "h2": ParagraphStyle("h2", parent=base["Normal"],
                             fontSize=14, textColor=ACCENT, fontName="Helvetica-Bold",
                             spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base["Normal"],
                             fontSize=10, textColor=TEAL, fontName="Helvetica-Bold",
                             spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontSize=9, textColor=TEXT, leading=14,
                               spaceAfter=6),
        "small": ParagraphStyle("small", parent=base["Normal"],
                                fontSize=7.5, textColor=colors.HexColor("#475569"),
                                leading=11),
        "center": ParagraphStyle("center", parent=base["Normal"],
                                 fontSize=9, alignment=TA_CENTER, textColor=TEXT),
        "label_pass": ParagraphStyle("lp", parent=base["Normal"],
                                     fontSize=8, textColor=GREEN,
                                     fontName="Helvetica-Bold"),
        "label_fail": ParagraphStyle("lf", parent=base["Normal"],
                                     fontSize=8, textColor=RED,
                                     fontName="Helvetica-Bold"),
        "code": ParagraphStyle("code", parent=base["Normal"],
                               fontSize=7.5, fontName="Courier",
                               textColor=colors.HexColor("#e2e8f0"),
                               backColor=NAVY, leftIndent=8, rightIndent=8,
                               leading=11, spaceBefore=4, spaceAfter=4),
    }
    return styles


# ── Main build ────────────────────────────────────────────────────────────────

def build_pdf(results_path: Path, output_path: Path) -> None:
    with open(results_path) as f:
        data = json.load(f)

    metrics = data["metrics"]
    results = data["results"]
    per_cat = metrics["per_category"]
    targets = metrics["targets_met"]

    date_str = datetime.now().strftime("%d %B %Y  %H:%M")
    title    = "Physics RAG Chatbot — Evaluation Report"

    rc = ReportCanvas(title, date_str)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=3.5*cm,  bottomMargin=2.2*cm,
    )

    S = get_styles()
    story = []

    # ── Cover splash ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.2*cm))
    story.append(Paragraph("PHYSICS RAG CHATBOT", S["h1"]))
    story.append(Paragraph("Evaluation Report", ParagraphStyle(
        "sub", parent=S["h1"], fontSize=13, textColor=TEAL, spaceAfter=4)))
    story.append(Paragraph(f"Generated {date_str}", ParagraphStyle(
        "date", parent=S["h1"], fontSize=9, textColor=MID_GRAY, spaceAfter=16)))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=16))

    # ── Executive summary boxes ───────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", S["h2"]))

    summary_data = [
        ["Metric",             "Score",   "Target",  "Status"],
        ["Citation Accuracy",  f"{metrics['citation_accuracy']:.1%}",
         "≥ 85%",  "PASS" if targets["citation_accuracy_85pct"] else "FAIL"],
        ["Hallucination Rate", f"{metrics['hallucination_rate']:.1%}",
         "< 10%",  "PASS" if targets["hallucination_below_10pct"] else "FAIL"],
        ["OOS Refusal Rate",   f"{metrics['refusal_rate']:.1%}",
         "≥ 90%",  "PASS" if targets["refusal_90pct"] else "FAIL"],
        ["Mean Confidence",    f"{metrics['mean_confidence']:.1%}", "—", "—"],
        ["Questions (total)",  str(metrics["total"]),               "—", "—"],
        ["Physics Qs",         str(metrics["physics_questions"]),   "—", "—"],
        ["OOS Qs",             str(metrics["out_of_scope_questions"]), "—", "—"],
    ]

    def status_color(val):
        if val == "PASS":  return colors.HexColor("#bbf7d0")
        if val == "FAIL":  return colors.HexColor("#fecaca")
        return LIGHT_GRAY

    tbl_style = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, WHITE]),
        ("FONTSIZE",     (0, 1), (-1, -1), 8.5),
        ("GRID",         (0, 0), (-1, -1), 0.4, MID_GRAY),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("ROUNDEDCORNERS", [4]),
    ])
    for i, row in enumerate(summary_data[1:], 1):
        status = row[3]
        tbl_style.add("BACKGROUND", (3, i), (3, i), status_color(status))
        if status == "PASS":
            tbl_style.add("TEXTCOLOR",   (3, i), (3, i), colors.HexColor("#166534"))
            tbl_style.add("FONTNAME",    (3, i), (3, i), "Helvetica-Bold")
        elif status == "FAIL":
            tbl_style.add("TEXTCOLOR",   (3, i), (3, i), colors.HexColor("#991b1b"))
            tbl_style.add("FONTNAME",    (3, i), (3, i), "Helvetica-Bold")

    tbl = Table(summary_data, colWidths=[5*cm, 3*cm, 3*cm, 3*cm])
    tbl.setStyle(tbl_style)
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    # All-pass banner
    all_pass = all(targets.values())
    banner_color = colors.HexColor("#bbf7d0") if all_pass else colors.HexColor("#fecaca")
    banner_text  = "ALL TARGETS MET — PRODUCTION READY" if all_pass else "SOME TARGETS NOT MET"
    banner_tc    = colors.HexColor("#166534") if all_pass else colors.HexColor("#991b1b")
    banner_tbl = Table([[banner_text]], colWidths=[14.4*cm])
    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), banner_color),
        ("TEXTCOLOR",    (0,0), (-1,-1), banner_tc),
        ("FONTNAME",     (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 10),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",   (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0), (-1,-1), 7),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Gauge charts ──────────────────────────────────────────────────────────
    story.append(Paragraph("Key Metrics", S["h2"]))

    gauge1 = make_gauge_chart(metrics["citation_accuracy"], 0.85,
                               "Citation Accuracy")
    gauge2 = make_gauge_chart(metrics["hallucination_rate"], 0.10,
                               "Hallucination Rate", above_is_good=False)
    gauge3 = make_gauge_chart(metrics["refusal_rate"], 0.90,
                               "OOS Refusal Rate")

    G = 4.5*cm
    gauge_tbl = Table([
        [Image(gauge1, width=G, height=G*0.65),
         Image(gauge2, width=G, height=G*0.65),
         Image(gauge3, width=G, height=G*0.65)],
    ], colWidths=[G+0.3*cm, G+0.3*cm, G+0.3*cm])
    gauge_tbl.setStyle(TableStyle([
        ("ALIGN",  (0,0),(-1,-1), "CENTER"),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0),(-1,-1), NAVY),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(gauge_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Category breakdown chart ──────────────────────────────────────────────
    story.append(Paragraph("Per-Category Breakdown", S["h2"]))
    cat_buf = make_category_bar(per_cat)
    story.append(Image(cat_buf, width=14.4*cm, height=5.6*cm))
    story.append(Spacer(1, 0.3*cm))

    # Category table
    cat_table_data = [["Category", "n", "Citation Acc", "Confidence", "Status"]]
    for cat, v in per_cat.items():
        acc  = v["citation_accuracy"]
        conf = v["mean_confidence"]
        if cat == "out_of_scope":
            status = "OOS"
        else:
            status = "PASS" if acc >= 0.85 else "FAIL"
        cat_table_data.append([
            cat.replace("_", " ").title(),
            str(v["count"]),
            f"{acc:.1%}",
            f"{conf:.1%}",
            status,
        ])

    cat_style = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  8.5),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (0, 0), (0, -1),  "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, WHITE]),
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("GRID",         (0, 0), (-1, -1), 0.4, MID_GRAY),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ])
    for i, row in enumerate(cat_table_data[1:], 1):
        s = row[4]
        if s == "PASS":
            cat_style.add("BACKGROUND", (4, i), (4, i), colors.HexColor("#bbf7d0"))
            cat_style.add("TEXTCOLOR",  (4, i), (4, i), colors.HexColor("#166534"))
            cat_style.add("FONTNAME",   (4, i), (4, i), "Helvetica-Bold")
        elif s == "FAIL":
            cat_style.add("BACKGROUND", (4, i), (4, i), colors.HexColor("#fecaca"))
            cat_style.add("TEXTCOLOR",  (4, i), (4, i), colors.HexColor("#991b1b"))
            cat_style.add("FONTNAME",   (4, i), (4, i), "Helvetica-Bold")

    cat_tbl = Table(cat_table_data,
                    colWidths=[4.5*cm, 1.2*cm, 3*cm, 3*cm, 2.7*cm])
    cat_tbl.setStyle(cat_style)
    story.append(cat_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Confidence scatter ────────────────────────────────────────────────────
    story.append(Paragraph("Per-Question Confidence", S["h2"]))
    sc_buf = make_confidence_scatter(results)
    story.append(Image(sc_buf, width=14.4*cm, height=5.4*cm))
    story.append(Spacer(1, 0.5*cm))

    # ── Page break before detailed results ───────────────────────────────────
    story.append(PageBreak())

    # ── Detailed Q&A table ────────────────────────────────────────────────────
    story.append(Paragraph("Detailed Question Results", S["h2"]))
    story.append(Paragraph(
        "Each row shows the benchmark question, model answer summary, "
        "citation accuracy, confidence score, and pass/fail status.",
        S["body"]))
    story.append(Spacer(1, 0.3*cm))

    qa_header = ["#", "Category", "Question (truncated)", "Cit. Acc", "Conf", "Status"]
    qa_data   = [qa_header]

    for r in results:
        q = r["question"][:55] + "…" if len(r["question"]) > 55 else r["question"]
        cat = r.get("category", "—").replace("_", " ").title()
        acc  = r.get("citation_accuracy", 0)
        conf = r.get("confidence", 0)

        if r.get("is_out_of_scope"):
            refused = r.get("correctly_refused", False)
            status = "REFUSED" if refused else "MISSED"
        else:
            status = "PASS" if acc >= 0.85 else "WARN"

        qa_data.append([
            str(r["id"]),
            cat,
            q,
            f"{acc:.0%}" if not r.get("is_out_of_scope") else "—",
            f"{conf:.0%}" if not r.get("is_out_of_scope") else "—",
            status,
        ])

    qa_style = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  8),
        ("FONTSIZE",     (0, 1), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, WHITE]),
        ("GRID",         (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (2, 0), (2, -1),  "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("WORDWRAP",     (2, 0), (2, -1),  True),
    ])
    for i, row in enumerate(qa_data[1:], 1):
        s = row[5]
        if s == "PASS":
            qa_style.add("BACKGROUND", (5,i),(5,i), colors.HexColor("#bbf7d0"))
            qa_style.add("TEXTCOLOR",  (5,i),(5,i), colors.HexColor("#166534"))
            qa_style.add("FONTNAME",   (5,i),(5,i), "Helvetica-Bold")
        elif s == "REFUSED":
            qa_style.add("BACKGROUND", (5,i),(5,i), colors.HexColor("#dbeafe"))
            qa_style.add("TEXTCOLOR",  (5,i),(5,i), colors.HexColor("#1e40af"))
            qa_style.add("FONTNAME",   (5,i),(5,i), "Helvetica-Bold")
        elif s in ("FAIL", "WARN", "MISSED"):
            qa_style.add("BACKGROUND", (5,i),(5,i), colors.HexColor("#fef9c3"))
            qa_style.add("TEXTCOLOR",  (5,i),(5,i), colors.HexColor("#854d0e"))
            qa_style.add("FONTNAME",   (5,i),(5,i), "Helvetica-Bold")

    qa_tbl = Table(qa_data,
                   colWidths=[0.7*cm, 2.6*cm, 6.5*cm, 1.4*cm, 1.2*cm, 2.0*cm])
    qa_tbl.setStyle(qa_style)
    story.append(qa_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Architecture section ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("System Architecture", S["h2"]))

    arch_text = (
        "The Physics RAG Chatbot implements a five-stage Retrieval-Augmented Generation "
        "pipeline grounded entirely in trusted textbook sources, eliminating LLM hallucination "
        "on undergraduate physics topics."
    )
    story.append(Paragraph(arch_text, S["body"]))
    story.append(Spacer(1, 0.3*cm))

    stages = [
        ("1. Ingestion", "PyMuPDF parses 6 PDFs (Feynman Lectures + OpenStax). Text is cleaned and split into 512-character chunks with 64-char overlap, preserving book/chapter/page metadata."),
        ("2. Embedding", "BAAI/bge-small-en-v1.5 (384-dim, local CPU) converts each chunk to a dense vector. Embedding runs once during ingestion and is stored persistently."),
        ("3. Retrieval", "MMR (Maximal Marginal Relevance) retrieves the top-10 diverse candidates from ChromaDB using cosine similarity."),
        ("4. Reranking", "cross-encoder/ms-marco-MiniLM-L-6-v2 scores all 10 candidates, returning the top-3 with a normalised confidence score (sigmoid of raw logits). Queries scoring below 0.35 are refused."),
        ("5. Generation", "Ollama Llama 3.1 (local) receives the system prompt, retrieved context as [Source N] blocks, and chat history. The Citation Engine maps [Source N] tags in the output back to book/chapter/page metadata."),
    ]

    for title_s, desc in stages:
        row_tbl = Table(
            [[Paragraph(f"<b>{title_s}</b>", ParagraphStyle(
                "st", parent=S["body"], textColor=ACCENT, fontName="Helvetica-Bold")),
              Paragraph(desc, S["body"])]],
            colWidths=[3.2*cm, 11.2*cm]
        )
        row_tbl.setStyle(TableStyle([
            ("VALIGN",      (0,0),(-1,-1), "TOP"),
            ("TOPPADDING",  (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING", (0,0),(0,-1),  8),
            ("LINEBELOW",   (0,0),(-1,-1), 0.3, LIGHT_GRAY),
        ]))
        story.append(row_tbl)

    story.append(Spacer(1, 0.5*cm))

    # ── Corpus table ──────────────────────────────────────────────────────────
    story.append(Paragraph("Corpus", S["h2"]))
    corpus_data = [
        ["Book", "Volumes", "Source", "Corpus ID"],
        ["The Feynman Lectures on Physics", "Vol 1 (Mechanics)\nVol 2 (E&M)\nVol 3 (QM)",
         "caltech.edu", "feynman"],
        ["OpenStax University Physics", "Vol 1 (Mechanics)\nVol 2 (Thermo/Waves)\nVol 3 (Optics/Modern)",
         "openstax.org", "openstax"],
    ]
    corpus_tbl = Table(corpus_data, colWidths=[5*cm, 4.5*cm, 3*cm, 2*cm])
    corpus_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0),  BLUE),
        ("TEXTCOLOR",    (0,0),(-1,0),  WHITE),
        ("FONTNAME",     (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1), 8),
        ("GRID",         (0,0),(-1,-1), 0.4, MID_GRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT_GRAY, WHITE]),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story.append(corpus_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Configuration ─────────────────────────────────────────────────────────
    story.append(Paragraph("Configuration Used", S["h2"]))
    config_data = [
        ["Parameter",               "Value"],
        ["LLM",                     "Llama 3.1 (Ollama, local)"],
        ["Embedding Model",         "BAAI/bge-small-en-v1.5 (384-dim)"],
        ["Reranker",                "cross-encoder/ms-marco-MiniLM-L-6-v2"],
        ["Vector Store",            "ChromaDB 1.0.12 (persistent)"],
        ["Chunk Size",              "512 characters / 64 overlap"],
        ["Retrieval Top-K",         "10 candidates → top-3 after reranking"],
        ["OOS Threshold",           "0.35 (reranker confidence)"],
        ["Embedding Device",        "CPU"],
        ["LLM Temperature",         "0.1"],
    ]
    cfg_tbl = Table(config_data, colWidths=[5.5*cm, 9*cm])
    cfg_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  BLUE),
        ("TEXTCOLOR",     (0,0),(-1,0),  WHITE),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTNAME",      (0,1),(0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("GRID",          (0,0),(-1,-1), 0.4, MID_GRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT_GRAY, WHITE]),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    story.append(cfg_tbl)

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story,
              onFirstPage=rc.on_page,
              onLaterPages=rc.on_page)

    print(f"\n  PDF report saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate PDF evaluation report")
    parser.add_argument("--input",  default="logs/evaluation_results.json",
                        help="Path to evaluation_results.json")
    parser.add_argument("--output", default="logs/physics_rag_report.pdf",
                        help="Output PDF path")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run evaluate.py first.")
        sys.exit(1)

    print(f"  Building PDF report from {in_path} ...")
    build_pdf(in_path, out_path)


if __name__ == "__main__":
    main()
