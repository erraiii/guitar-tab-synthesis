import numpy as np


class FusionProcessor:
    def __init__(self, mapper):
        self.mapper = mapper

    # --- 1. ФИЛЬТР ПО РУКЕ ---
    def _filter_by_hand(self, audio_positions, fingering):
        """
        Оставляем только:
        - открытые струны
        - лады в окрестности руки
        """
        if not fingering:
            return audio_positions

        # frets = [f for _, f in fingering]
        frets = [pos.fret for pos in fingering.positions]
        hand_min = min(frets)
        hand_max = max(frets)

        filtered = []

        for s, f in audio_positions:
            if f == 0:
                filtered.append((s, f))
            elif hand_min - 1 <= f <= hand_max + 1:
                filtered.append((s, f))

        return filtered

    # --- 2. СКОРИНГ ---
    def _score_position(self, pos, visual_candidates):
        """
        Чем ближе к визуалу - тем лучше
        """
        s, f = pos

        # идеальное совпадение
        if pos in visual_candidates:
            return 100

        penalties = []

        for vs, vf in visual_candidates:
            if s == vs:
                penalties.append(abs(f - vf))

        if penalties:
            return -min(penalties)

        # если струна вообще не наблюдалась
        return -10

    # --- 3. FUSE ДЛЯ ОДНОЙ НОТЫ ---
    def fuse_note(self, note, fingering, visual_candidates):
        """
        note.midi -> (string, fret)
        """

        # 1. аудио кандидаты
        audio_positions = self.mapper.get_positions(note.pitch)

        if not audio_positions:
            return None

        # 2. фильтрация по руке
        filtered = self._filter_by_hand(audio_positions, fingering)

        if not filtered:
            filtered = audio_positions  # fallback

        # 3. выбираем лучший по скорингу
        best = max(
            filtered,
            key=lambda pos: self._score_position(pos, visual_candidates)
        )

        return best

    # --- 4. FUSE ДЛЯ СОБЫТИЯ ---
    def fuse_event(self, event, fingering, visual_candidates):
        result = []

        for i, note in enumerate(event.notes):
            pos = self.fuse_note(note, fingering, visual_candidates)
            if pos is not None:
                result.append(pos)

        return result