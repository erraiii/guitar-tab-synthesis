import numpy as np
from config import FUSION_PERFECT_SCORE, FUSION_UNSEEN_PENALTY


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
            return FUSION_PERFECT_SCORE

        penalties = []

        for vs, vf in visual_candidates:
            if s == vs:
                penalties.append(abs(f - vf))

        if penalties:
            return -min(penalties)

        # если струна вообще не наблюдалась
        return FUSION_UNSEEN_PENALTY

    # --- 4. FUSE ДЛЯ СОБЫТИЯ ---
    def fuse_event(self, event, fingering, visual_candidates):
        """
        Теперь учитывает, что струна может быть использована только один раз
        """

        candidates = []

        # 1. собираем все кандидаты
        for note_idx, note in enumerate(event.notes):
            audio_positions = self.mapper.get_positions(note.pitch)

            filtered = self._filter_by_hand(audio_positions, fingering)
            if not filtered:
                filtered = audio_positions

            for pos in filtered:
                score = self._score_position(pos, visual_candidates)
                candidates.append((score, note_idx, pos))

        # 2. сортируем по убыванию score
        candidates.sort(reverse=True, key=lambda x: x[0])

        used_strings = set()
        used_notes = set()
        result = [None] * len(event.notes)

        # 3. жадное назначение
        for score, note_idx, pos in candidates:
            string, fret = pos

            if note_idx in used_notes:
                continue
            if string in used_strings:
                continue

            result[note_idx] = pos
            used_notes.add(note_idx)
            used_strings.add(string)

        # 4. удаляем None (если не удалось назначить)
        result = [r for r in result if r is not None]

        return result