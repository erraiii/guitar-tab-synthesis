from core.models import FingerPosition, FingeringFrame
from geometry.region import point_to_region


class FingeringProcessor:
    def detect(self, fingertips, fret_lines, string_lines, timestamp):
        positions = []

        for pt in fingertips:
            string_id, fret_id = point_to_region(
                pt, fret_lines, string_lines
            )

            if string_id is not None and fret_id is not None:
                positions.append(FingerPosition(string_id, fret_id))

        return FingeringFrame(timestamp, positions)