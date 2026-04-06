import numpy as np
from .primitives import compute_mean_direction, compute_line_pts, fit_line
from .primitives import (filter_by_hands, remove_duplicate_frets,
                        align_string_direction)

class GeometryProcessor:
    """
        hand: {"box": ..., "fingertips": [...]}
        guitar_det: GuitarDetections

        return:
            список (string_id, fret_id)
    """

    def process(self, hand, guitar_det):

        frets = guitar_det.frets
        # 1. фильтрация по руке
        active_frets, rejected_frets = filter_by_hands(frets, hand)

        # 2. одно направление
        fret_lines = align_string_direction(active_frets)

        # 3. средний угол
        angle = compute_mean_direction(fret_lines)

        # 4. точки струн
        for f in active_frets:
            f.string_points = compute_line_pts(np.array(f.center_line))

        # 5. сбор точек по струнам
        str_points = [
            [f.string_points[i] for f in active_frets]
            for i in range(7)
        ]

        # 6. линии между струнами
        string_regions = [fit_line(points) for points in str_points]
        results = string_regions


        return results