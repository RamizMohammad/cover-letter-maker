"""Render the (possibly hand-edited) letter text to a one-page-style A4 PDF."""
import datetime
import io
import unicodedata
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

_REPLACE = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ", "•": "-",
}


def _clean(s: str) -> str:
    """Built-in PDF fonts only cover Latin-1; fold everything else down safely."""
    for k, v in _REPLACE.items():
        s = s.replace(k, v)
    out = []
    for ch in s:
        try:
            ch.encode("latin-1")
            out.append(ch)
        except UnicodeEncodeError:
            out.append(unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode())
    return "".join(out)


def render(letter: str, name: str = "", contact: list[str] | None = None) -> bytes:
    ink, muted = HexColor("#1a1a1a"), HexColor("#666666")
    name_style = ParagraphStyle("name", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=ink)
    contact_style = ParagraphStyle("contact", fontName="Helvetica", fontSize=9, leading=13, textColor=muted)
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15.5, textColor=ink, spaceAfter=10)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=25 * mm, rightMargin=25 * mm, topMargin=22 * mm, bottomMargin=22 * mm,
        title="Cover Letter", author=_clean(name),
    )
    flow = []
    if name.strip():
        flow.append(Paragraph(escape(_clean(name.strip())), name_style))
    bits = [escape(_clean(c.strip())) for c in (contact or []) if c and c.strip()]
    if bits:
        flow.append(Paragraph("  |  ".join(bits), contact_style))
    if flow:
        flow += [Spacer(1, 6), HRFlowable(width="100%", thickness=0.6, color=HexColor("#cccccc")), Spacer(1, 14)]

    today = datetime.date.today()
    flow += [Paragraph(f"{today.day} {today.strftime('%B %Y')}", body), Spacer(1, 4)]

    for para in _clean(letter).replace("\r\n", "\n").split("\n\n"):
        para = para.strip()
        if para:
            flow.append(Paragraph(escape(para).replace("\n", "<br/>"), body))
    doc.build(flow)
    return buf.getvalue()
