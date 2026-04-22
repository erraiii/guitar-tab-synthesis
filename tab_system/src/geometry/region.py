import logging

logger = logging.getLogger(__name__)


def line_val(line, point):
    a, b, c = line
    x, y = point
    return a*x + b*y + c


def find_fret(point, fret_lines):
    for i in range(len(fret_lines)-1):
        d1 = line_val(fret_lines[i], point)
        d2 = line_val(fret_lines[i+1], point)

        if d1 * d2 <= 0:
            return i + 1
    return None


def find_string(point, string_lines):
    for i in range(len(string_lines)-1):
        d1 = line_val(string_lines[i], point)
        d2 = line_val(string_lines[i+1], point)

        if d1 * d2 <= 0:
            return i + 1
    return None


def point_to_region(point, fret_lines, string_lines):
    fret_id = find_fret(point, fret_lines)
    string_id = find_string(point, string_lines)
    return string_id, fret_id