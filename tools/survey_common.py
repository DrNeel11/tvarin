"""Build docs/Ramer_Literature_Survey.pdf.

A literature survey of local-LLM inference optimization methods, with an
empirical positioning of this repository's method (Ramer) against them.

Every measured number attributed to this project is traceable to
PRODUCT_HYPOTHESES.md, docs/ARCHITECTURE.md, or results/. Every external
claim is attributed to a numbered reference.

    python tools/survey_figures.py     # first, writes docs/figures/*.png
    python tools/build_survey_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "Ramer_Literature_Survey.pdf"

TITLE = ("Optimizing Local Large Language Model Inference: A Survey of Methods "
         "and an Empirical Case for Measurement-Based Placement")
AUTHOR = "Neel"
DATE = "24 September 2026"

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#3a3a38")
MUTED = colors.HexColor("#6f6e6a")
RULE = colors.HexColor("#c9c8c4")
ACCENT = colors.HexColor("#2a78d6")
BAND = colors.HexColor("#f3f5f8")
BAND2 = colors.HexColor("#fbfbfa")

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()

body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Times-Roman",
                      fontSize=10, leading=14.2, alignment=TA_JUSTIFY,
                      spaceBefore=0, spaceAfter=7, textColor=INK)
body_first = ParagraphStyle("body_first", parent=body, firstLineIndent=0)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=3,
                        spaceAfter=4)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                    fontSize=13.5, leading=16.5, spaceBefore=17, spaceAfter=7,
                    textColor=INK)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=11, leading=14, spaceBefore=12, spaceAfter=5,
                    textColor=INK)
h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Oblique",
                    fontSize=10, leading=13, spaceBefore=9, spaceAfter=3,
                    textColor=INK2)
cap = ParagraphStyle("cap", parent=body, fontName="Helvetica", fontSize=8.2,
                     leading=11.2, alignment=TA_LEFT, textColor=MUTED,
                     spaceBefore=4, spaceAfter=12)
tcell = ParagraphStyle("tcell", parent=body, fontName="Helvetica", fontSize=7.6,
                       leading=9.6, alignment=TA_LEFT, spaceAfter=0,
                       textColor=INK2)
thead = ParagraphStyle("thead", parent=tcell, fontName="Helvetica-Bold",
                       textColor=INK)
ref = ParagraphStyle("ref", parent=body, fontSize=8.6, leading=11.2,
                     leftIndent=17, firstLineIndent=-17, spaceAfter=4.5,
                     alignment=TA_LEFT)
abstract = ParagraphStyle("abstract", parent=body, fontSize=9.4, leading=13.2,
                          leftIndent=16, rightIndent=16)
title_s = ParagraphStyle("title", parent=ss["Title"], fontName="Times-Bold",
                         fontSize=19, leading=23.5, spaceAfter=8, textColor=INK)
sub_s = ParagraphStyle("sub", parent=ss["Normal"], fontName="Times-Italic",
                       fontSize=11.5, leading=15, alignment=TA_CENTER,
                       textColor=INK2)
meta_s = ParagraphStyle("meta", parent=ss["Normal"], fontName="Helvetica",
                        fontSize=9.5, leading=13, alignment=TA_CENTER,
                        textColor=MUTED)
note_s = ParagraphStyle("note", parent=body, fontSize=8.8, leading=12,
                        leftIndent=10, rightIndent=10, textColor=INK2)


# ---------------------------------------------------------------- helpers
def P(text, style=body):
    return Paragraph(text, style)


def B(text):
    return Paragraph(text, bullet, bulletText="•")


def N(text, num):
    return Paragraph(text, bullet, bulletText=f"{num}.")


def H1(text, key):
    p = Paragraph(text, h1)
    p._toc = (0, text, key)
    return p


def H2(text, key):
    p = Paragraph(text, h2)
    p._toc = (1, text, key)
    return p


def H3(text):
    return Paragraph(text, h3)


def figure(name, caption, width=None):
    path = FIG / name
    if not path.exists():
        raise SystemExit(f"missing figure {path}; run tools/survey_figures.py first")
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    w = width or 15.6 * cm
    img = Image(str(path), width=w, height=w * ih / iw)
    img.hAlign = "CENTER"
    return [img, P(caption, cap)]


def table(rows, widths, caption=None, header=True, font=7.6, align=None):
    data = []
    for r_i, row in enumerate(rows):
        out = []
        for c_i, cell in enumerate(row):
            st = thead if (header and r_i == 0) else tcell
            if font != 7.6:
                st = ParagraphStyle(f"t{font}{r_i}{c_i}", parent=st, fontSize=font,
                                    leading=font * 1.28)
            out.append(Paragraph(str(cell), st))
        data.append(out)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, INK),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND2]),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
    ]
    if align:
        for col, a in align.items():
            style.append(("ALIGN", (col, 0), (col, -1), a))
    t.setStyle(TableStyle(style))
    items = [t]
    if caption:
        items.append(P(caption, cap))
    return items


def callout(text):
    t = Table([[Paragraph(text, note_s)]], colWidths=[15.6 * cm], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return [t, Spacer(1, 9)]


# ---------------------------------------------------------------- canvas
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, A4[1] - 1.35 * cm,
                      "Optimizing Local LLM Inference: A Survey")
    canvas.drawRightString(A4[0] - doc.rightMargin, A4[1] - 1.35 * cm, DATE)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, A4[1] - 1.55 * cm,
                A4[0] - doc.rightMargin, A4[1] - 1.55 * cm)
    canvas.drawCentredString(A4[0] / 2.0, 1.25 * cm, str(canvas.getPageNumber()))
    canvas.restoreState()


def on_title(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0] / 2.0, 1.25 * cm, "1")
    canvas.restoreState()


class Doc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        toc = getattr(flowable, "_toc", None)
        if toc:
            level, text, key = toc
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, text, self.page, key))


def build(story):
    doc = Doc(str(OUT), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
              topMargin=2.1 * cm, bottomMargin=2.0 * cm, title=TITLE, author=AUTHOR,
              subject="Literature survey of local LLM inference optimization")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([
        PageTemplate(id="title", frames=[frame], onPage=on_title),
        PageTemplate(id="main", frames=[frame], onPage=on_page),
    ])
    doc.multiBuild(story)
    print("wrote", OUT)
