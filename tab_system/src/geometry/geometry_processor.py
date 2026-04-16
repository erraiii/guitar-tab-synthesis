import numpy as np
from .primitives import compute_mean_direction, compute_line_pts, fit_line, frets_to_abc, point_dir_to_abc, \
    sort_frets_right_to_left, sort_strings_bottom_to_top
from .primitives import filter_by_hands, align_string_direction

class GeometryProcessor:
    """
        hand: {"box": ..., "fingertips": [...]}
        guitar_det: GuitarDetections

        return:
            список (string_id, fret_id)
    """

    def process(self, hand, guitar_det, shape):
        if guitar_det is None or not guitar_det.frets:
            return [], []

        frets = guitar_det.frets
        # 1. фильтрация по руке
        active_frets, rejected_frets = filter_by_hands(frets, hand)

        if len(active_frets) < 2:
            return [], []

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
        string_regions = sort_strings_bottom_to_top(string_regions, shape)

        # 7. линии порожков
        # 7.1 линии для порожков, прошедших фильтр
        fret_regions = frets_to_abc(active_frets, angle)

        # 7.2 линии отфильтрованных порожков и верхнего порожка
        if guitar_det.nut is not None:
            rejected_frets = rejected_frets + [guitar_det.nut]
        fret_reg_rej = [point_dir_to_abc(fret.center, angle) for fret in rejected_frets]

        fret_regions.extend(fret_reg_rej)
        fret_regions = sort_frets_right_to_left(fret_regions, shape)

        return string_regions, fret_regions