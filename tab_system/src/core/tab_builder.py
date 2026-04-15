STANDARD_TUNING_NAMES = ["E", "B", "G", "D", "A", "E"]  # 1 → 6


class TabBuilder:
    def __init__(self, capo=None, max_cols=40):
        self.capo = capo or 0
        self.max_cols = max_cols

        # 6 струн
        self.strings = [[] for _ in range(6)]

    def _empty_column(self):
        return ["-" for _ in range(6)]

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

        for i in range(6):
            val = column[i]
            if val == "-":
                column[i] = "-" * width
            else:
                column[i] = val.rjust(width, "-")

        # добавляем в струны
        for i in range(6):
            self.strings[i].append(column[i])

        for i in range(6):
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

        # return "\n\n".join(chunks)