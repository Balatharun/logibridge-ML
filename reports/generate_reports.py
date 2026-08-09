"""
Generate phase1_report.pdf and phase2_report.pdf from markdown sources.
Uses fpdf2 (pure Python, no GTK/Pango required on Windows).
Run: python reports/generate_reports.py
"""
import re
from pathlib import Path
from fpdf import FPDF

REPORTS = Path(__file__).parent

BLUE  = (26, 58, 92)
WHITE = (255, 255, 255)
LGREY = (247, 250, 252)
DGREY = (100, 100, 100)


class ReportPDF(FPDF):
    def __init__(self, title):
        super().__init__()
        self.report_title = title.replace("\u2014", "-").replace("\u2013", "-")
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()
        self.set_margins(20, 20, 20)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*DGREY)
        self.cell(0, 8, self.report_title, align="R")
        self.ln(2)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.3)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DGREY)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def render_md_to_pdf(md_path: Path, pdf_path: Path):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    # sanitise to latin-1 safe characters
    lines = [l.replace("\u2014", "-").replace("\u2013", "-")
               .replace("\u2019", "'").replace("\u2018", "'")
               .replace("\u201c", '"').replace("\u201d", '"')
               .replace("\u2192", "->").replace("\u2190", "<-")
               .replace("\u2022", "-").replace("\u00b7", "-")
               .encode("latin-1", errors="replace").decode("latin-1")
             for l in lines]
    title = next((l.lstrip("# ") for l in lines if l.startswith("# ")), md_path.stem)
    pdf = ReportPDF(title)
    pdf.set_font("Helvetica", "", 10)

    in_code = False
    in_table = False
    table_rows = []
    table_header = []

    def flush_table():
        if not table_header:
            return
        col_w = (pdf.w - 40) / max(len(table_header), 1)
        # header row
        pdf.set_fill_color(*BLUE)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 8)
        for h in table_header:
            pdf.cell(col_w, 7, h[:30], border=1, fill=True)
        pdf.ln()
        # data rows
        pdf.set_text_color(0, 0, 0)
        for i, row in enumerate(table_rows):
            pdf.set_fill_color(*(LGREY if i % 2 == 0 else WHITE))
            pdf.set_font("Helvetica", "", 8)
            for cell in row:
                pdf.cell(col_w, 6, str(cell)[:30], border=1, fill=True)
            pdf.ln()
        pdf.ln(2)
        table_header.clear()
        table_rows.clear()

    for line in lines:
        # code block
        if line.startswith("```"):
            if in_table:
                flush_table()
                in_table = False
            in_code = not in_code
            if in_code:
                pdf.set_font("Courier", "", 8)
                pdf.set_fill_color(241, 245, 249)
            else:
                pdf.set_font("Helvetica", "", 10)
                pdf.ln(2)
            continue

        if in_code:
            pdf.set_x(20)
            pdf.multi_cell(170, 5, line, fill=True)
            continue

        # table rows
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.match(r"^[-: ]+$", c) for c in cells if c):
                continue  # separator row
            if not table_header:
                table_header.extend(cells)
                in_table = True
            else:
                table_rows.append(cells)
            continue
        elif in_table:
            flush_table()
            in_table = False

        # headings
        if line.startswith("#### "):
            pdf.set_x(20)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*BLUE)
            pdf.multi_cell(170, 6, line[5:])
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("### "):
            pdf.ln(2)
            pdf.set_x(20)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*BLUE)
            pdf.multi_cell(170, 7, line[4:])
            pdf.set_draw_color(200, 200, 200)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        elif line.startswith("## "):
            pdf.ln(3)
            pdf.set_x(20)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_fill_color(*BLUE)
            pdf.set_text_color(*WHITE)
            pdf.cell(170, 9, "  " + line[3:], fill=True)
            pdf.ln(5)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("# "):
            pdf.set_x(20)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(*BLUE)
            pdf.multi_cell(170, 10, line[2:])
            pdf.set_draw_color(*BLUE)
            pdf.set_line_width(0.5)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(4)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("---"):
            pdf.set_draw_color(200, 200, 200)
            pdf.line(20, pdf.get_y() + 2, 190, pdf.get_y() + 2)
            pdf.ln(4)
        elif line.strip() == "":
            pdf.ln(2)
        else:
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            text = re.sub(r"`(.+?)`", r"\1", text)
            text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
            pdf.set_x(20)
            if line.startswith("- ") or line.startswith("* "):
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(170, 5, "  - " + text[2:])
            else:
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(170, 5, text)

    if in_table:
        flush_table()

    pdf.output(str(pdf_path))
    print(f"  Saved: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    for md_name, pdf_name in [("phase1_report.md", "phase1_report.pdf"),
                               ("phase2_report.md", "phase2_report.pdf")]:
        print(f"Converting {md_name} ...")
        render_md_to_pdf(REPORTS / md_name, REPORTS / pdf_name)
    print("\nDone. Reports saved to reports/")
