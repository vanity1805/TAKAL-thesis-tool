"""Write a report (see ``report_data.build_report``) as an organised Excel workbook.

Layout: a "Read me" sheet with a clickable contents list, one summary sheet per
section (key figures, readings, tables, native charts), a separate sheet for the
long weekly table (so its header can stay frozen), and a Glossary sheet. Numbers are
stored as real numbers with number formats, so they can still be sorted and used in
calculations.

Colour is kept for the table headers and the charts. Everything else uses a
neutral ink-on-white scheme with hairline rules.
"""

from __future__ import annotations

import io
import math
from datetime import date

import numpy as np
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from report_data import SECTION_BLURBS

FOREST = "0D2B24"   # table headers
TEXT = "1B2220"    # body text
MUTED = "5E6967"   # secondary text
RULE = "D8DCDB"    # hairlines and outlines
RULE_DARK = "8E9794"  # closing line of a table, emphasis rules
PANEL = "F4F5F5"   # the only fill: reading boxes and summary rows
FONT = "Calibri"

FORMATS = {
    "date": "mmm d, yyyy",
    "peso": '"₱"#,##0.00',
    "peso_signed": '"+₱"#,##0.00;"−₱"#,##0.00;"₱"0.00',
    "dec3": "0.000",
    "dec3_signed": "+0.000;−0.000;0.000",
    "dec4": "0.0000",
    "dec6": "0.000000",
    "pct0": "0%",
    "pct1": '0.0"%"',
    "int": "0",
}

thin = Side(style="thin", color=RULE)
dark = Side(style="thin", color=RULE_DARK)
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
HAIR = Border(bottom=thin)


def _font(**kw):
    return Font(name=FONT, **kw)


def _fill(hex_):
    return PatternFill("solid", start_color=hex_, end_color=hex_)


def _clean(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _align_for(kind: str) -> str:
    """Every table column, header and cell, is centred so the tables read uniformly."""
    return "center"


def _banner(ws, title: str, subtitle: str, width: int) -> int:
    """Sheet title: dark ink on white with a rule under it, no colour fill."""
    ws.sheet_view.showGridLines = False
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=width)
    cell = ws.cell(1, 1, title)
    cell.font = _font(size=18, bold=True, color=TEXT)
    cell.alignment = Alignment(vertical="center", indent=1)
    ws.row_dimensions[1].height = 34
    for c in range(1, width + 1):
        ws.cell(1, c).border = Border(bottom=Side(style="medium", color=TEXT))
    ws.row_dimensions[2].height = 4
    if not subtitle:  # no subtitle line: just a small gap before the content
        ws.row_dimensions[3].height = 10
        return 4
    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=width)
    sub = ws.cell(3, 1, subtitle)
    sub.font = _font(size=10, color=MUTED)
    sub.alignment = Alignment(wrap_text=True, vertical="top", indent=1)
    ws.row_dimensions[3].height = max(18, 15 * math.ceil(len(subtitle) / (width * 14)))
    return 5


def _paragraph(ws, row: int, text: str, width: int, *, bold=False, italic=False, color=TEXT,
               fill: str | None = None, size=10, chars_per_col=14) -> int:
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=width)
    cell = ws.cell(row, 1, text)
    cell.font = _font(size=size, bold=bold, italic=italic, color=color)
    cell.alignment = Alignment(wrap_text=True, vertical="top", indent=1)
    if fill:
        for c in range(1, width + 1):
            ws.cell(row, c).fill = _fill(fill)
    lines = max(1, math.ceil(len(text) / max(20, width * chars_per_col)))
    ws.row_dimensions[row].height = max(18, 14.5 * lines + 4)
    return row + 1


def _heading(ws, row: int, text: str, width: int) -> int:
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=width)
    cell = ws.cell(row, 1, text)
    cell.font = _font(size=12.5, bold=True, color=TEXT)
    cell.alignment = Alignment(vertical="center", indent=1)
    ws.row_dimensions[row].height = 24
    for c in range(1, width + 1):
        ws.cell(row, c).border = Border(bottom=dark)
    return row + 1


def _cards(ws, row: int, items: list, width: int) -> int:
    span = max(1, width // len(items))
    for i, (label, value, detail) in enumerate(items):
        c0 = 1 + i * span
        c1 = c0 + span - 1 if i < len(items) - 1 else width
        for offset, (text, font) in enumerate((
            (label, _font(size=9, color=MUTED)),
            (value, _font(size=15, bold=True, color=TEXT)),
            (detail, _font(size=8.5, color=MUTED)),
        )):
            ws.merge_cells(start_row=row + offset, start_column=c0, end_row=row + offset, end_column=c1)
            cell = ws.cell(row + offset, c0, text)
            cell.font = font
            cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            for c in range(c0, c1 + 1):
                ws.cell(row + offset, c).fill = _fill("FFFFFF")
                ws.cell(row + offset, c).border = Border(
                    left=thin if c == c0 else None, right=thin if c == c1 else None,
                    top=thin if offset == 0 else None,
                    bottom=thin if offset == 2 else None,
                )
    ws.row_dimensions[row].height = 30
    ws.row_dimensions[row + 1].height = 26
    ws.row_dimensions[row + 2].height = 30
    return row + 4


def _write_table(ws, row: int, block: dict, width: int, col_widths: dict[int, float]) -> tuple[int, int, int]:
    """Write a titled table; return (next_row, header_row, first_data_row)."""
    df, kinds = block["df"], block["kinds"]
    row = _heading(ws, row, block["title"], width)
    if block.get("note"):
        row = _paragraph(ws, row, block["note"], width, italic=True, color=MUTED, size=9)
    header_row = row
    for j, column in enumerate(df.columns, start=1):
        cell = ws.cell(row, j, column)
        cell.font = _font(size=10, bold=True, color="FFFFFF")
        cell.fill = _fill(FOREST)
        horizontal = _align_for(kinds[column])
        cell.alignment = Alignment(horizontal=horizontal, vertical="center", wrap_text=True,
                                   indent=1 if horizontal != "center" else 0)
        cell.border = Border(left=Side(style="thin", color="1B473A"), right=Side(style="thin", color="1B473A"))
        col_widths[j] = max(col_widths.get(j, 10), min(40, len(str(column)) + 4))
    ws.row_dimensions[row].height = 32
    first = row + 1
    last = first + len(df) - 1
    for i in range(len(df)):
        r = first + i
        overall = str(df.iloc[i, 0]) == "Overall"
        for j, column in enumerate(df.columns, start=1):
            kind = kinds[column]
            raw = df.iloc[i, j - 1]
            value = raw if kind == "date" else _clean(raw)
            if kind == "date" and hasattr(raw, "year"):
                value = date(raw.year, raw.month, raw.day)
            cell = ws.cell(r, j, value if value is not None else "Not yet")
            # hairline between rows, a darker line to close the table, a rule above the summary row
            cell.border = Border(
                bottom=dark if i == len(df) - 1 else thin,
                top=dark if overall else None,
            )
            cell.font = _font(size=10, bold=overall, color=TEXT)
            if kind in FORMATS and value is not None:
                cell.number_format = FORMATS[kind]
            horizontal = _align_for(kind)
            cell.alignment = Alignment(horizontal=horizontal, vertical="center", wrap_text=kind in ("text", "effect"),
                                       indent=1 if horizontal != "center" else 0)
            if overall:
                cell.fill = _fill(PANEL)
            if kind == "peso_signed" and isinstance(value, float):
                cell.font = _font(size=10, bold=True, color="B45309" if value > 0.005 else "0369A1" if value < -0.005 else MUTED)
            if kind == "effect":
                text = str(value).lower()
                cell.font = _font(size=10, bold=True, color="0B8561" if text.startswith("raises") else "0369A1" if text.startswith("lowers") else MUTED)
            if value is None or value == "Not yet":
                cell.font = _font(size=10, italic=True, color=MUTED)
            shown = len(f"{value:,.4f}") if isinstance(value, float) else len(str(value))
            col_widths[j] = max(col_widths.get(j, 10), min(46, shown + 3))
    return first + len(df) + 1, header_row, first


def _outline(ws, r0: int, r1: int, width: int) -> None:
    """Draw a thin outline around rows r0..r1 across the first ``width`` columns."""
    for r in range(r0, r1 + 1):
        for c in range(1, width + 1):
            ws.cell(r, c).border = Border(
                left=thin if c == 1 else None, right=thin if c == width else None,
                top=thin if r == r0 else None, bottom=thin if r == r1 else None,
            )


def _reading(ws, row: int, title: str, lines: list[str], width: int) -> int:
    """A light-gray outlined box: neutral, with no colour accent."""
    start = row
    row = _paragraph(ws, row, title, width, bold=True, color=TEXT, fill=PANEL, size=10.5)
    for i, line in enumerate(lines):
        row = _paragraph(ws, row, line, width, italic=(i == 0), color=MUTED if i == 0 else TEXT, fill=PANEL, size=10)
    _outline(ws, start, row - 1, width)
    return row + 1


def _apply_widths(ws, col_widths: dict[int, float], width: int) -> None:
    for j in range(1, width + 1):
        ws.column_dimensions[get_column_letter(j)].width = max(col_widths.get(j, 14), 22)


def _soften(chart) -> None:
    """Light-gray gridlines so the data, not the grid, carries the chart."""
    grid = getattr(chart.y_axis, "majorGridlines", None)
    if grid is not None:
        grid.spPr = GraphicalProperties(ln=LineProperties(solidFill="E3E6E5"))


def _sheet_name(base: str, taken: set[str]) -> str:
    name = base[:31]
    taken.add(name)
    return name


def build_excel(report: dict) -> bytes:
    wb = Workbook()
    readme = wb.active
    readme.title = "Read me"
    taken = {"Read me"}
    links: list[tuple[str, str, str]] = []

    for number, section in enumerate(report["sections"], start=1):
        key = section["key"]
        ws = wb.create_sheet(_sheet_name(f"{number}. {key}", taken))
        tables = [b for b in section["blocks"] if b["type"] == "table"]
        width = max([len(b["df"].columns) for b in tables] + [6])
        col_widths: dict[int, float] = {}
        row = _banner(ws, f"Section {number} · {section['title']}", section["intro"], width)
        links.append((ws.title, f"Section {number} · {section['title']}", SECTION_BLURBS[key]))

        long_sheet = None
        data_ref = None
        for block in section["blocks"]:
            if block["type"] == "cards":
                row = _cards(ws, row, block["items"], width)
            elif block["type"] == "heading":
                row = _heading(ws, row, block["text"], width)
                row = _paragraph(ws, row, block["caption"], width, italic=True, color=MUTED, size=9)
                row += 1
            elif block["type"] == "chart":
                continue  # native charts are added after the tables so they can reference cells
            elif block["type"] == "table" and block["id"] == "overview_weekly":
                # long table lives on its own sheet with a frozen header
                long_sheet = wb.create_sheet(_sheet_name(f"{number}. Weekly table", taken))
                lw = len(block["df"].columns)
                lcw: dict[int, float] = {}
                lrow = _banner(long_sheet, "Weekly table", f"{block['title']}. {block.get('note', '')}", lw)
                lrow, header_row, first = _write_table(long_sheet, lrow, {**block, "title": "Weekly forecast versus actual", "note": ""}, lw, lcw)
                long_sheet.freeze_panes = long_sheet.cell(first, 1)
                long_sheet.auto_filter.ref = f"A{header_row}:{get_column_letter(lw)}{first + len(block['df']) - 1}"
                for j in range(1, lw + 1):
                    long_sheet.column_dimensions[get_column_letter(j)].width = max(lcw.get(j, 14), 18)
                data_ref = (long_sheet, header_row, first, len(block["df"]))
                links.append((long_sheet.title, f"Section {number} · Weekly table (all test weeks)", "Every week with forecast, actual and difference. Header stays in view while scrolling."))
                row = _paragraph(ws, row, f"The full week-by-week table is on the sheet “{long_sheet.title}”.", width, italic=True, color=MUTED, size=9.5)
                row += 1
                row = _reading(ws, row, block["reading_title"], block["reading"], width)
            elif block["type"] == "table":
                row, header_row, first = _write_table(ws, row, block, width, col_widths)
                if block["id"] == "importance":
                    data_ref_imp = (header_row, first, len(block["df"]))
                    imp_ref = data_ref_imp
                row = _reading(ws, row, block["reading_title"], block["reading"], width)

        # ---- native charts ----
        for block in section["blocks"]:
            if block["type"] != "chart":
                continue
            if block["kind"] == "forecast" and data_ref:
                sheet, header_row, first, n = data_ref
                chart = LineChart()
                chart.title = block["title"]
                chart.y_axis.title = "PHP per kg"
                chart.height, chart.width = 8.5, 24
                for col in (2, 3):
                    chart.add_data(Reference(sheet, min_col=col, min_row=header_row, max_row=first + n - 1), titles_from_data=True)
                chart.set_categories(Reference(sheet, min_col=1, min_row=first, max_row=first + n - 1))
                chart.x_axis.number_format = "mmm yyyy"
                chart.x_axis.delete = False
                chart.y_axis.delete = False
                values = [v for v in [*block["actual"], *next(iter(block["series"].values()))] if v is not None and v == v]
                chart.y_axis.scaling.min = math.floor(min(values)) - 1
                chart.y_axis.scaling.max = math.ceil(max(values)) + 1
                chart.series[0].graphicalProperties.line.solidFill = "1BB385"
                chart.series[1].graphicalProperties.line.solidFill = "000000"
                for s in chart.series:
                    s.smooth = False
                    s.marker.symbol = "none"
                _soften(chart)
                ws.add_chart(chart, f"A{row}")
                row += 19
                row = _paragraph(ws, row, "The green line is the forecast and the black line is the actual price.", width, italic=True, color=MUTED, size=9)
            elif block["kind"] == "difference" and data_ref:
                sheet, header_row, first, n = data_ref
                chart = BarChart()
                chart.type = "col"
                chart.title = block["title"]
                chart.y_axis.title = "Forecast − Actual (PHP/kg)"
                chart.height, chart.width = 7.5, 24
                chart.add_data(Reference(sheet, min_col=4, min_row=header_row, max_row=first + n - 1), titles_from_data=True)
                chart.set_categories(Reference(sheet, min_col=1, min_row=first, max_row=first + n - 1))
                chart.legend = None
                chart.x_axis.number_format = "mmm yyyy"
                chart.x_axis.delete = False
                chart.y_axis.delete = False
                chart.series[0].graphicalProperties.solidFill = "B45309"
                chart.series[0].invertIfNegative = False
                _soften(chart)
                ws.add_chart(chart, f"A{row}")
                row += 17
                row = _paragraph(ws, row, block["caption"], width, italic=True, color=MUTED, size=9)
            elif block["kind"] == "importance":
                header_row, first, n = imp_ref
                chart = BarChart()
                chart.type = "bar"
                chart.title = block["title"]
                chart.height, chart.width = 7.5, 20
                chart.add_data(Reference(ws, min_col=3, min_row=header_row, max_row=first + n - 1), titles_from_data=True)
                chart.set_categories(Reference(ws, min_col=2, min_row=first, max_row=first + n - 1))
                chart.legend = None
                chart.y_axis.title = "Average impact (PHP per kg)"
                chart.x_axis.scaling.orientation = "maxMin"
                chart.x_axis.delete = False
                chart.y_axis.delete = False
                chart.series[0].graphicalProperties.solidFill = "0EA5E9"
                _soften(chart)
                ws.add_chart(chart, f"A{row}")
                row += 17
                row = _paragraph(ws, row, block["caption"], width, italic=True, color=MUTED, size=9)
        _apply_widths(ws, col_widths, width)
        ws.column_dimensions["A"].width = max(col_widths.get(1, 14), 16)
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True

    # ---- glossary ----
    gl = wb.create_sheet("Glossary")
    taken.add("Glossary")
    row = _banner(gl, "Glossary", "Plain-language meaning of the terms used in this workbook.", 2)
    for j, text in enumerate(("Term", "What it means"), start=1):
        cell = gl.cell(row, j, text)
        cell.font = _font(bold=True, color="FFFFFF")
        cell.fill = _fill(FOREST)
        cell.alignment = Alignment(vertical="center", indent=1)
        cell.border = Border(left=Side(style="thin", color="1B473A"), right=Side(style="thin", color="1B473A"))
    gl.row_dimensions[row].height = 24
    total = len(report["glossary"])
    for i, (term, meaning) in enumerate(report["glossary"], start=1):
        a, b = gl.cell(row + i, 1, term), gl.cell(row + i, 2, meaning)
        a.font, b.font = _font(bold=True, color=TEXT), _font(color=TEXT)
        for c in (a, b):
            c.border = Border(bottom=dark if i == total else thin)
            c.alignment = Alignment(wrap_text=True, vertical="top", indent=1)
        gl.row_dimensions[row + i].height = max(18, 15 * math.ceil(len(meaning) / 95))
    gl.column_dimensions["A"].width = 28
    gl.column_dimensions["B"].width = 100
    links.append(("Glossary", "Glossary", "Plain-language meaning of every term used in this workbook."))

    # ---- read me ----
    rm = readme
    row = _banner(rm, report["title"], "", 3)  # the title stands alone, no subtitle line
    for key, value in report["info"]:
        rm.cell(row, 1, key).font = _font(bold=True, color=MUTED)
        rm.merge_cells(start_row=row, start_column=2, end_row=row, end_column=3)
        rm.cell(row, 2, value).font = _font(color=TEXT)
        rm.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="top")
        rm.cell(row, 1).alignment = Alignment(vertical="top", indent=1)
        rm.row_dimensions[row].height = max(18, 15 * math.ceil(len(value) / 70))
        for c in (1, 2, 3):
            rm.cell(row, c).border = Border(bottom=thin)
        row += 1
    row += 1
    row = _heading(rm, row, "What is inside (click a name to jump to it)", 3)
    for sheet_name, label, blurb in links:
        cell = rm.cell(row, 1, label)
        cell.hyperlink = f"#'{sheet_name}'!A1"
        cell.font = _font(bold=True, color="0369A1", underline="single")
        rm.merge_cells(start_row=row, start_column=2, end_row=row, end_column=3)
        rm.cell(row, 2, blurb).font = _font(color=MUTED)
        rm.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="top")
        rm.cell(row, 1).alignment = Alignment(wrap_text=True, vertical="top", indent=1)
        rm.row_dimensions[row].height = max(18, 15 * math.ceil(len(blurb) / 70))
        for c in (1, 2, 3):
            rm.cell(row, c).border = Border(bottom=thin)
        row += 1
    row += 1
    row = _heading(rm, row, "How to read this workbook", 3)
    for line in (
        "Each section sheet starts with a short summary and key figures, followed by tables and charts.",
        "Under every table, a shaded “Reading” box explains in plain words what the numbers mean.",
        "Prices are in Philippine pesos (₱) per kilogram. Cells hold real numbers, so you can sort and calculate with them.",
    ):
        row = _paragraph(rm, row, "• " + line, 3, chars_per_col=40)
    if report.get("data_note"):
        row += 1
        row = _heading(rm, row, "Data note", 3)
        row = _paragraph(rm, row, report["data_note"], 3, color=MUTED, chars_per_col=40)
    rm.column_dimensions["A"].width = 44
    rm.column_dimensions["B"].width = 40
    rm.column_dimensions["C"].width = 40

    for sheet in wb.worksheets:
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.sheet_properties.tabColor = RULE_DARK
        sheet.oddFooter.left.text = "TAKAL · Rice Price Forecast Report"
        sheet.oddFooter.left.size = 8
        sheet.oddFooter.right.text = "Page &P of &N"
        sheet.oddFooter.right.size = 8

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()