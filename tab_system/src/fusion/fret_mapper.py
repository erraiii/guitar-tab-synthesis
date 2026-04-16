from config import STANDARD_TUNING, MAX_FRETS


class FretboardMapper:
    """Маппер MIDI нот в позиции на грифе гитары"""

    def __init__(self, tuning=STANDARD_TUNING):
        """
        Инициализация маппера

        Args:
            tuning: список MIDI нот для открытых струн (от нижней к верхней)
                    Если не указан, используется стандартный строй
        """
        self.tuning = tuning
        self.num_strings = len(self.tuning)

        # Полная lookup таблица для всех MIDI нот (0-127)
        self.lookup = {note: [] for note in range(128)}
        self._build_lookup()

    def _build_lookup(self):
        """Предвычисляет все позиции на грифе для всех MIDI нот"""
        for string_idx, open_note in enumerate(self.tuning):
            # Номер струны для пользователя (1 = самая верхняя, 6 = самая нижняя)
            string_number = self.num_strings - string_idx

            for fret in range(MAX_FRETS):  # 0-24 лады
                midi_note = open_note + fret

                if 0 <= midi_note <= 127:
                    self.lookup[midi_note].append((string_number, fret))

    def get_positions(self, midi_note):
        """
        Возвращает все возможные позиции для заданной MIDI ноты

        Args:
            midi_note: MIDI номер ноты (0-127)

        Returns:
            список кортежей (string_number, fret)
            string_number: 1-6 (1 - самая верхняя, 6 - самая нижняя)
            fret: 0-24 (0 - открытая струна)
        """
        if not 0 <= midi_note <= 127:
            return []
        return self.lookup.get(midi_note, [])

    def get_best_position(self, midi_note, preferred_string=None):
        """
        Возвращает "лучшую" позицию

        Args:
            midi_note: MIDI номер ноты
            preferred_string: предпочитаемая струна (1-6)

        Returns:
            кортеж (string_number, fret) или None
        """
        positions = self.get_positions(midi_note)
        if not positions:
            return None

        if preferred_string is not None:
            for pos in positions:
                if pos[0] == preferred_string:
                    return pos

        return min(positions, key=lambda x: x[1])