"""Draw a report (see ``report_data.build_report``) as a readable PDF."""

from __future__ import annotations

import html
import io
import os
from datetime import date, datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from interpretation import peso

# ---- palette (matches the dashboard) ---------------------------------------
FOREST = colors.HexColor("#0D2B24")
GREEN = colors.HexColor("#10B981")
MINT = colors.HexColor("#E9F8F3")
BORDER = colors.HexColor("#D5E0DB")
MUTED = colors.HexColor("#5F746D")
TEXT = colors.HexColor("#0F1F1C")
ZEBRA = colors.HexColor("#F5F8F6")
UP = "#B45309"     # forecast above actual
DOWN = "#0369A1"   # forecast below actual

PAGE_W, PAGE_H = A4
MARGIN = 17 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ---- fonts (DejaVu ships with matplotlib and has the peso sign, minus, R²) ----
def _register_fonts() -> None:
    if "TakalSans" in pdfmetrics.getRegisteredFontNames():
        return
    font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("TakalSans", os.path.join(font_dir, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("TakalSans-Bold", os.path.join(font_dir, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("TakalSans-Italic", os.path.join(font_dir, "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily(
        "TakalSans", normal="TakalSans", bold="TakalSans-Bold", italic="TakalSans-Italic", boldItalic="TakalSans-Bold"
    )


def _styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName="TakalSans", textColor=TEXT)
    return {
        "body": ParagraphStyle("body", fontSize=9.5, leading=14, spaceAfter=4, **base),
        "muted": ParagraphStyle("muted", fontSize=8.5, leading=12, **{**base, "textColor": MUTED}),
        "title": ParagraphStyle("title", fontSize=26, leading=31, fontName="TakalSans-Bold", textColor=FOREST),
        "subtitle": ParagraphStyle("subtitle", fontSize=11, leading=15, **{**base, "textColor": MUTED}),
        "h_banner": ParagraphStyle("h_banner", fontSize=17, leading=21, fontName="TakalSans-Bold", textColor=colors.white),
        "h_banner_sub": ParagraphStyle("h_banner_sub", fontSize=9, leading=12, fontName="TakalSans", textColor=colors.HexColor("#CFE3DA")),
        "h2": ParagraphStyle("h2", fontSize=12, leading=16, fontName="TakalSans-Bold", textColor=FOREST, spaceBefore=10, spaceAfter=4),
        "caption": ParagraphStyle("caption", fontSize=8, leading=11, fontName="TakalSans-Italic", textColor=MUTED, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontSize=8.2, leading=10.5, alignment=TA_LEFT, **base),
        "cell_c": ParagraphStyle("cell_c", fontSize=8.2, leading=10.5, alignment=TA_CENTER, **base),
        "th": ParagraphStyle("th", fontSize=8.2, leading=10.5, alignment=TA_CENTER, fontName="TakalSans-Bold", textColor=colors.white),
        "card_label": ParagraphStyle("card_label", fontSize=7.8, leading=10, **{**base, "textColor": MUTED}),
        "card_value": ParagraphStyle("card_value", fontSize=13.5, leading=17, fontName="TakalSans-Bold", textColor=FOREST),
        "card_detail": ParagraphStyle("card_detail", fontSize=7.4, leading=9.5, **{**base, "textColor": MUTED}),
        "read_title": ParagraphStyle("read_title", fontSize=9.2, leading=13, fontName="TakalSans-Bold", textColor=FOREST),
        "read_how": ParagraphStyle("read_how", fontSize=8.6, leading=12.5, fontName="TakalSans-Italic", textColor=MUTED, spaceBefore=3),
        "read": ParagraphStyle("read", fontSize=8.8, leading=12.8, spaceBefore=3, **base),
        "toc": ParagraphStyle("toc", fontSize=10, leading=15, **base),
    }


# ---- value formatting --------------------------------------------------------
def fmt(kind: str, value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "Not yet" if kind in ("peso", "peso_signed") else "—"
    if kind == "date":
        return value.strftime("%b %d, %Y").replace(" 0", " ") if isinstance(value, (date, datetime)) else str(value)
    if kind == "peso":
        return peso(float(value))
    if kind == "peso_signed":
        return peso(float(value), signed=True)
    if kind == "dec3":
        return f"{float(value):,.3f}"
    if kind == "dec3_signed":
        return f"{float(value):+.3f}".replace("-", "−")
    if kind == "dec4":
        return f"{float(value):.4f}".replace("-", "−")
    if kind == "dec6":
        return f"{float(value):.6f}"
    if kind == "pct0":
        return f"{float(value):.0%}"
    if kind == "pct1":
        return f"{float(value):.1f}%"
    if kind == "int":
        return str(int(value))
    return str(value)


def _esc(text) -> str:
    return html.escape(str(text))


# ---- charts --------------------------------------------------------------------
def _png(fig) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9D6D0")
    ax.tick_params(colors="#3E554D", labelsize=8)
    ax.grid(axis="y", color="#E6EDE9", linewidth=0.7)
    ax.set_axisbelow(True)


def _dates(weeks):
    return [datetime.strptime(w, "%Y-%m-%d") for w in weeks]


def _chart_forecast(block: dict) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    x = _dates(block["weeks"])
    palette = {"multivariate": "#1BB385", "univariate": "#6E998B"}
    names = {"multivariate": "Multivariate forecast", "univariate": "Univariate forecast"}
    for model, values in block["series"].items():
        active = model == block["active"]
        ax.plot(x, values, color=palette[model], lw=2.2 if active else 1.6,
                ls="-" if active else (0, (2, 2)), label=names[model], zorder=3 if active else 2)
    ax.plot(x, block["actual"], color="black", lw=2.0, label="Actual price", zorder=4)
    ax.set_ylabel("PHP per kg", fontsize=8.5, color="#3E554D")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    _style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=8)
    return _png(fig)


def _chart_difference(block: dict) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    x = _dates(block["weeks"])
    values = np.asarray(block["values"], dtype=float)
    ax.bar(x, np.nan_to_num(values), width=5, color=[UP if v > 0 else DOWN for v in np.nan_to_num(values)])
    ax.axhline(0, color="#3E554D", lw=0.9)
    ax.set_ylabel("Forecast − Actual (PHP/kg)", fontsize=8.5, color="#3E554D")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    _style_axes(ax)
    return _png(fig)


def _chart_importance(block: dict) -> io.BytesIO:
    labels, values = block["labels"][::-1], block["values"][::-1]
    fig, ax = plt.subplots(figsize=(7.4, max(2.2, 0.5 * len(labels) + 0.6)))
    color = "#0EA5E9" if block["model"] == "multivariate" else "#64748B"
    bars = ax.barh(labels, values, color=color)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"₱{value:.2f}", va="center", fontsize=8, color="#0F1F1C")
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("Average impact on the forecast (PHP per kg)", fontsize=8.5, color="#3E554D")
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#E6EDE9", linewidth=0.7)
    return _png(fig)


def _chart_waterfall(block: dict) -> io.BytesIO:
    labels, values = block["labels"], block["values"]
    base = block["baseline"]
    fig, ax = plt.subplots(figsize=(7.4, max(2.4, 0.46 * len(labels) + 0.9)))
    running = base
    for i, value in enumerate(values):
        ax.barh(i, value, left=running, color="#10B981" if value >= 0 else "#0EA5E9", height=0.6)
        ax.text(max(running, running + value) + 0.05, i, f"{value:+.2f}".replace("-", "−"),
                va="center", fontsize=8, color="#0F1F1C")
        running += value
    ax.axvline(base, color="#64748B", ls="--", lw=1)
    ax.axvline(block["predicted"], color="#0EA5E9", lw=1.2)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Running estimate (PHP per kg)   |   baseline ₱{base:.2f} (dashed) → forecast ₱{block['predicted']:.2f} (solid)",
                  fontsize=8.2, color="#3E554D")
    lo = min([base, block["predicted"]] + [base + sum(values[: k + 1]) for k in range(len(values))])
    hi = max([base, block["predicted"]] + [base + sum(values[: k + 1]) for k in range(len(values))])
    ax.set_xlim(lo - 0.6, hi + 0.9)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#E6EDE9", linewidth=0.7)
    return _png(fig)


CHARTS = {
    "forecast": _chart_forecast,
    "difference": _chart_difference,
    "importance": _chart_importance,
    "waterfall": _chart_waterfall,
}


# ---- flowables -----------------------------------------------------------------
def _banner(number: int, title: str, intro: str, st: dict) -> list:
    inner = Table(
        [[Paragraph(f"SECTION {number}", st["h_banner_sub"])], [Paragraph(_esc(title), st["h_banner"])]],
        colWidths=[CONTENT_W - 16],
    )
    inner.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    banner = Table([[inner]], colWidths=[CONTENT_W])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), FOREST),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, 0), (-1, -1), 3, GREEN),
    ]))
    return [banner, Spacer(1, 7), Paragraph(_esc(intro), st["body"]), Spacer(1, 4)]


def _cards(items: list, st: dict) -> Table:
    width = CONTENT_W / len(items)
    cells = [[
        [Paragraph(_esc(label), st["card_label"]), Paragraph(_esc(value), st["card_value"]),
         Paragraph(_esc(detail), st["card_detail"])]
        for label, value, detail in items
    ]]
    table = Table(cells, colWidths=[width] * len(items))
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER), ("INNERGRID", (0, 0), (-1, -1), 0.6, BORDER),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEABOVE", (0, 0), (-1, 0), 2.2, GREEN),
    ]))
    return table


def _column_widths(df, kinds: dict, texts: dict[str, list[str]]) -> list[float]:
    weights = []
    for column in df.columns:
        longest = max([len(str(column))] + [len(s) for s in texts[column]])
        cap = 34 if kinds[column] in ("text", "effect") else 16
        weights.append(max(7, min(longest, cap)) + 3)
    total = sum(weights)
    return [CONTENT_W * w / total for w in weights]


def _table(block: dict, st: dict) -> Table:
    df, kinds = block["df"], block["kinds"]
    columns = list(df.columns)
    texts = {c: [fmt(kinds[c], v) for v in df[c]] for c in columns}
    rows = [[Paragraph(_esc(c), st["th"]) for c in columns]]
    for i in range(len(df)):
        row = []
        for c in columns:
            kind, text, raw = kinds[c], texts[c][i], df[c].iloc[i]
            if kind in ("text", "effect"):
                if kind == "effect":
                    tone = ("#0B8561" if text.lower().startswith("raises")
                            else "#0369A1" if text.lower().startswith("lowers") else "#5F746D")
                    row.append(Paragraph(f'<font color="{tone}"><b>{_esc(text)}</b></font>', st["cell_c"]))
                else:
                    row.append(Paragraph(_esc(text), st["cell"]))
            elif kind == "peso_signed" and isinstance(raw, (int, float, np.floating)) and np.isfinite(raw):
                colour = UP if raw > 0.005 else DOWN if raw < -0.005 else "#5F746D"
                row.append(Paragraph(f'<font color="{colour}"><b>{_esc(text)}</b></font>', st["cell_c"]))
            else:
                row.append(Paragraph(_esc(text), st["cell_c"]))
        rows.append(row)
    table = Table(rows, colWidths=_column_widths(df, kinds, texts), repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), FOREST), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, BORDER), ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in range(2, len(rows), 2):
        style.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    first = texts[columns[0]]
    for r, label in enumerate(first, start=1):  # emphasise the 'Overall' row
        if str(label) == "Overall":
            style += [("BACKGROUND", (0, r), (-1, r), MINT), ("LINEABOVE", (0, r), (-1, r), 1, GREEN)]
    table.setStyle(TableStyle(style))
    return table


def _reading_box(title: str, lines: list[str], st: dict) -> Table:
    content = [Paragraph(_esc(title), st["read_title"])]
    for i, line in enumerate(lines):
        content.append(Paragraph(_esc(line), st["read_how"] if i == 0 else st["read"]))
    box = Table([[content]], colWidths=[CONTENT_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), MINT), ("LINEBEFORE", (0, 0), (0, -1), 3.5, GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return box


def _section_flow(number: int, section: dict, st: dict) -> list:
    flow = _banner(number, section["title"], section["intro"], st)
    for block in section["blocks"]:
        kind = block["type"]
        if kind == "cards":
            flow += [_cards(block["items"], st), Spacer(1, 8)]
        elif kind == "chart":
            image = Image(CHARTS[block["kind"]](block))
            image.drawWidth = CONTENT_W
            image.drawHeight = CONTENT_W * image.imageHeight / image.imageWidth
            flow.append(KeepTogether([
                Paragraph(_esc(block["title"]), st["h2"]), image, Paragraph(_esc(block["caption"]), st["caption"]),
            ]))
        elif kind == "heading":
            flow.append(KeepTogether([
                Spacer(1, 6), Paragraph(_esc(block["text"]), st["h2"]),
                Paragraph(_esc(block["caption"]), st["caption"]),
            ]))
        elif kind == "table":
            head = [Paragraph(_esc(block["title"]), st["h2"])]
            if block.get("note"):
                head.append(Paragraph(_esc(block["note"]), st["caption"]))
            table = _table(block, st)
            reading = _reading_box(block["reading_title"], block["reading"], st)
            if len(block["df"]) <= 16:
                flow.append(KeepTogether(head + [table, Spacer(1, 6), reading, Spacer(1, 8)]))
            else:
                flow += [CondPageBreak(70 * mm)] + head + [table, Spacer(1, 8), reading, Spacer(1, 8)]
    return flow


def _footer(label: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)
        canvas.setFont("TakalSans", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 8.5 * mm, "TAKAL · Rice Price Forecast Report")
        canvas.drawRightString(PAGE_W - MARGIN, 8.5 * mm, f"{label}   ·   Page {doc.page}")
        canvas.restoreState()
    return draw


def _load_logo(path: str | None):
    """Crop transparent margins and recolor white logo text to dark forest green for the white PDF cover."""
    if not path or not os.path.exists(path):
        return None
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as source:
            rgba = source.convert("RGBA")
            bbox = rgba.getchannel("A").getbbox()
            if bbox:
                rgba = rgba.crop(bbox)

            arr = np.array(rgba, dtype=np.uint8)
            rgb = arr[..., :3].astype(int)
            alpha = arr[..., 3]

            # Identify colored icon pixels (salakot hat & blue measuring cup)
            chroma = rgb.max(axis=2) - rgb.min(axis=2)
            colored = (alpha > 40) & (chroma > 18)

            # White rice & cup interior are enclosed on all 4 sides by colored hat/cup pixels
            has_left = np.maximum.accumulate(colored, axis=1)
            has_right = np.maximum.accumulate(colored[:, ::-1], axis=1)[:, ::-1]
            has_top = np.maximum.accumulate(colored, axis=0)
            has_bottom = np.maximum.accumulate(colored[::-1, :], axis=0)[::-1, :]
            enclosed = colored | (has_left & has_right & has_top & has_bottom)

            # Include dark outline pixels immediately bordering the hat/cup
            h, w = enclosed.shape
            near_enclosed = np.zeros_like(enclosed)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    y0, y1 = max(0, dy), h + min(0, dy)
                    x0, x1 = max(0, dx), w + min(0, dx)
                    near_enclosed[y0:y1, x0:x1] |= enclosed[y0 - dy : y1 - dy, x0 - dx : x1 - dx]

            brightness = rgb.max(axis=2)
            icon_mask = enclosed | (near_enclosed & (brightness < 160))

            # Recolor all non-icon visible pixels ("AKAL", arrow, subtitle) to TAKAL forest (#0D2B24)
            text_mask = (~icon_mask) & (alpha > 0)
            arr[text_mask, 0] = 13
            arr[text_mask, 1] = 43
            arr[text_mask, 2] = 36

            recolored = PILImage.fromarray(arr, mode="RGBA")
            white_bg = PILImage.new("RGBA", recolored.size, (255, 255, 255, 255))
            composited = PILImage.alpha_composite(white_bg, recolored).convert("RGB")
            composited.thumbnail((1400, 520), PILImage.Resampling.LANCZOS)

            buffer = io.BytesIO()
            composited.save(buffer, format="PNG")

        buffer.seek(0)
        logo = Image(buffer)
        ratio = logo.imageWidth / logo.imageHeight
        logo.drawHeight = 24 * mm
        logo.drawWidth = 24 * mm * ratio
        logo.hAlign = "CENTER"
        return logo
    except Exception:  # a missing or unreadable logo must never block the export
        return None


def build_pdf(report: dict, logo_path: str | None = None) -> bytes:
    _register_fonts()
    st = _styles()
    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm, bottomMargin=19 * mm,
        title=report["title"], author="TAKAL",
    )
    frame = Frame(MARGIN, 19 * mm, CONTENT_W, PAGE_H - 35 * mm, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    templates = [PageTemplate(id="cover", frames=[frame], onPage=_footer("Cover"))]
    for section in report["sections"]:
        templates.append(PageTemplate(id=section["key"], frames=[frame], onPage=_footer(section["title"])))
    templates.append(PageTemplate(id="Glossary", frames=[frame], onPage=_footer("Glossary")))
    doc.addPageTemplates(templates)

    # ---- cover ----
    story: list = []
    logo = _load_logo(logo_path)
    if logo is not None:
        story += [logo, Spacer(1, 10)]
    story += [Paragraph(_esc(report["title"]), st["title"]), Spacer(1, 4),
              Paragraph(_esc(report["subtitle"]), st["subtitle"]), Spacer(1, 16)]

    info = Table([[Paragraph(f"<b>{_esc(k)}</b>", st["cell"]), Paragraph(_esc(v), st["cell"])] for k, v in report["info"]],
                 colWidths=[40 * mm, CONTENT_W - 40 * mm])
    info.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, BORDER), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [info, Spacer(1, 16), Paragraph("What is inside", st["h2"])]

    from report_data import SECTION_BLURBS
    for number, section in enumerate(report["sections"], start=1):
        story.append(Paragraph(f"<b>Section {number} — {_esc(section['title'])}</b><br/>"
                               f"<font color='#5F746D' size='9'>{_esc(SECTION_BLURBS[section['key']])}</font>", st["toc"]))
        story.append(Spacer(1, 3))
    story.append(Paragraph("<b>Glossary</b><br/><font color='#5F746D' size='9'>Plain-language meaning of every term used in this report.</font>", st["toc"]))

    story += [Spacer(1, 14), Paragraph("How to read this report", st["h2"]),
              Paragraph("Each section opens with a short summary of what it shows. Every table is followed by a "
                        "green <b>Reading</b> box that explains, in plain words, what the numbers mean. "
                        "Prices are in Philippine pesos (₱) per kilogram.", st["body"])]
    if report.get("data_note"):
        story += [Spacer(1, 6), Paragraph("Data note", st["h2"]), Paragraph(_esc(report["data_note"]), st["muted"])]

    for number, section in enumerate(report["sections"], start=1):
        story += [NextPageTemplate(section["key"]), PageBreak()]
        story += _section_flow(number, section, st)

    story += [NextPageTemplate("Glossary"), PageBreak()]
    story += _banner(len(report["sections"]) + 1, "Glossary", "Plain-language meaning of the terms used in this report.", st)
    glossary = Table([[Paragraph(f"<b>{_esc(t)}</b>", st["cell"]), Paragraph(_esc(d), st["cell"])] for t, d in report["glossary"]],
                     colWidths=[42 * mm, CONTENT_W - 42 * mm], repeatRows=0)
    glossary.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, BORDER), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(glossary)

    doc.build(story)
    return buffer.getvalue()