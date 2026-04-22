import logging

logger = logging.getLogger(__name__)


def generate_visual_candidates(finger_positions, capo=None, num_strings=6):
    """
    Генерирует кандидаты позиций

    return:
        [(string, fret), ...]
    """

    open_fret = capo if capo is not None else 0

    if not finger_positions:
        finger_positions = []

    # собираем ВСЕ лады по струнам
    string_to_frets = {}
    for pos in finger_positions:
        s = pos.string
        f = pos.fret
        string_to_frets.setdefault(s, []).append(f)

    result = []

    for string in range(1, num_strings + 1):
        # всегда открытая
        result.append((string, open_fret))

        # все пальцы на струне
        if string in string_to_frets:
            for fret in string_to_frets[string]:
                result.append((string, fret))

    return result