from config import TAB_MAX_COLS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging

logger = logging.getLogger(__name__)

STANDARD_TUNING_NAMES = ["E", "B", "G", "D", "A", "E"]  # 1 → 6
NUM_STRINGS = 6


class TabBuilder:
    def __init__(self, capo=None, max_cols=None):
        self.capo = capo or 0
        self.max_cols = max_cols if max_cols is not None else TAB_MAX_COLS

        # 6 струн
        self.strings = [[] for _ in range(NUM_STRINGS)]

    def _empty_column(self):
        return ["-" for _ in range(NUM_STRINGS)]

    def add_event(self, fused_positions):
        """
        fused_positions: [(string, fret)]
        """
        column = self._empty_column()

        for string, fret in fused_positions:
            real_fret = fret - self.capo

            if real_fret < 0:
                continue

            idx = string - 1
            column[idx] = str(real_fret)

        # выравнивание по ширине (чтобы 10 не ломал таб)
        width = max(len(x) for x in column)

        for i in range(NUM_STRINGS):
            val = column[i]
            if val == "-":
                column[i] = "-" * width
            else:
                column[i] = val.rjust(width, "-")

        # добавляем в струны
        for i in range(NUM_STRINGS):
            self.strings[i].append(column[i])

        for i in range(NUM_STRINGS):
            self.strings[i].append("---")

    def render(self):
        lines = []

        # каподастр
        if self.capo > 0:
            lines.append(f"Capo fret {self.capo}")

        # строй
        for i, name in enumerate(STANDARD_TUNING_NAMES):
            line = name + "|" + "---"
            line += "".join(self.strings[i])
            lines.append(line)

        return "\n".join(lines)

    def render_chunked(self):
        """
        Делит табы на блоки по max_cols
        """
        chunks = []

        total = len(self.strings[0])

        for start in range(0, total, self.max_cols):
            end = start + self.max_cols

            block = []

            for i, name in enumerate(STANDARD_TUNING_NAMES):
                line = name + "|" + "---"
                line += "".join(self.strings[i][start:end])
                block.append(line)

            chunks.append("\n".join(block))

        result = []

        if self.capo > 0:
            result.append(f"Capo fret {self.capo}")

        result.append("\n\n".join(chunks))

        return "\n".join(result)


def save_tabs_pdf(content: str, output_path, lines_per_page: int = 60):
    """
    Сохраняет табулатуру в PDF в моноширинном виде.
    """
    lines = content.splitlines()
    pages = max(1, (len(lines) + lines_per_page - 1) // lines_per_page)

    with PdfPages(output_path) as pdf:
        for page_idx in range(pages):
            start = page_idx * lines_per_page
            end = start + lines_per_page
            chunk = lines[start:end]
            page_text = "\n".join(chunk)

            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.text(
                0.05,
                0.98,
                page_text,
                va="top",
                ha="left",
                family="monospace",
                fontsize=8
            )
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)