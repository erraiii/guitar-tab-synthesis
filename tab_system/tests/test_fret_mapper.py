# test_fretboard_mapper.py
from fusion.fret_mapper import FretboardMapper, STANDARD_TUNING, map_event


class MockNote:
    """Мок-класс для ноты"""

    def __init__(self, midi):
        self.midi = midi


class MockEvent:
    """Мок-класс для события"""

    def __init__(self, midi_notes):
        self.notes = [MockNote(midi) for midi in midi_notes]


class TestFretboardMapper:
    """Тесты для FretboardMapper"""

    def test_initialization_with_default_tuning(self):
        """Тест инициализации со стандартным строем"""
        mapper = FretboardMapper()
        assert mapper.tuning == STANDARD_TUNING
        assert mapper.num_strings == 6
        assert len(mapper.lookup) == 128

    def test_initialization_with_custom_tuning(self):
        """Тест инициализации с кастомным строем"""
        custom_tuning = [38, 45, 50, 55, 59, 64]  # Drop D
        mapper = FretboardMapper(custom_tuning)
        assert mapper.tuning == custom_tuning
        assert mapper.num_strings == 6

    def test_get_positions_open_strings(self):
        """Тест открытых струн"""
        mapper = FretboardMapper()

        # E2 (6-я открытая)
        positions = mapper.get_positions(40)
        assert (6, 0) in positions

        # A2 (5-я открытая)
        positions = mapper.get_positions(45)
        assert (5, 0) in positions

        # E4 (1-я открытая)
        positions = mapper.get_positions(64)
        assert (1, 0) in positions

    def test_get_positions_fretted_notes(self):
        """Тест зажатых нот"""
        mapper = FretboardMapper()

        # 5-й лад на 6-й струне = A2 (45)
        positions = mapper.get_positions(45)
        assert (6, 5) in positions

        # 5-й лад на 5-й струне = D3 (50)
        positions = mapper.get_positions(50)
        assert (5, 5) in positions

        # 12-й лад на 1-й струне = E5 (76)
        positions = mapper.get_positions(76)
        assert (1, 12) in positions

    def test_get_positions_multiple_positions(self):
        """Тест ноты, которая может быть сыграна на разных струнах"""
        mapper = FretboardMapper()

        # E4 (64) может быть: открытая 1-я струна, 5-й лад на 2-й, 9-й лад на 3-й и т.д.
        positions = mapper.get_positions(64)

        assert (1, 0) in positions  # открытая 1-я
        assert (2, 5) in positions  # 5-й лад на 2-й струне
        assert (3, 9) in positions  # 9-й лад на 3-й струне
        assert (4, 14) in positions  # 14-й лад на 4-й струне
        assert (5, 19) in positions  # 19-й лад на 5-й струне
        assert (6, 24) in positions  # 24-й лад на 6-й струне

    def test_get_positions_out_of_range(self):
        """Тест нот вне диапазона MIDI"""
        mapper = FretboardMapper()

        # MIDI вне диапазона 0-127
        positions = mapper.get_positions(-1)
        assert positions == []

        positions = mapper.get_positions(128)
        assert positions == []

    def test_get_positions_note_not_on_guitar(self):
        """Тест ноты, которой нет на гитаре"""
        mapper = FretboardMapper()

        # MIDI 10 - очень низкая нота, недоступна на стандартной гитаре
        positions = mapper.get_positions(10)
        assert positions == []

        # MIDI 120 - очень высокая нота
        positions = mapper.get_positions(120)
        assert positions == []

    def test_get_positions_all_notes_in_range(self):
        """Тест, что все MIDI ноты от 0 до 127 покрыты"""
        mapper = FretboardMapper()

        for midi in range(128):
            positions = mapper.get_positions(midi)
            assert isinstance(positions, list)  # всегда список

    def test_best_position_no_preference(self):
        """Тест выбора лучшей позиции без предпочтений"""
        mapper = FretboardMapper()

        # E4 имеет несколько позиций, должна вернуться первая
        best = mapper.get_best_position(64)
        assert best is not None
        assert isinstance(best, tuple)
        assert len(best) == 2

    def test_best_position_with_preference(self):
        """Тест выбора лучшей позиции с предпочтением по струне"""
        mapper = FretboardMapper()

        # Предпочитаем 2-ю струну для E4
        best = mapper.get_best_position(64, preferred_string=2)
        assert best == (2, 5)  # 5-й лад на 2-й струне

    def test_best_position_not_found(self):
        """Тест выбора позиции для недоступной ноты"""
        mapper = FretboardMapper()

        best = mapper.get_best_position(10)
        assert best is None

    def test_best_position_preference_not_available(self):
        """Тест предпочтения недоступной струны"""
        mapper = FretboardMapper()

        best = mapper.get_best_position(65, preferred_string=6)
        assert best == (1, 1)  # возвращается первая доступная позиция


    def test_no_duplicate_notes_in_string(self):
        """Тест отсутствия дубликатов для одной струны"""
        mapper = FretboardMapper()

        for string_idx, open_note in enumerate(mapper.tuning):
            seen_notes = set()
            for fret in range(25):
                midi_note = open_note + fret
                if 0 <= midi_note <= 127:
                    # Одна и та же нота не должна встречаться дважды на одной струне
                    assert midi_note not in seen_notes
                    seen_notes.add(midi_note)


class TestMapEvent:
    """Тесты для функции map_event"""

    def test_map_event_with_default_mapper(self):
        """Тест маппинга события с маппером по умолчанию"""
        event = MockEvent([40, 45, 64])
        result = map_event(event)

        assert len(result) == 3
        assert (6, 0) in result[0]
        assert (5, 0) in result[1]
        assert (1, 0) in result[2]


    def test_map_event_with_custom_mapper(self):
        """Тест маппинга события с кастомным маппером"""
        custom_tuning = [38, 45, 50, 55, 59, 64]  # Drop D
        mapper = FretboardMapper(custom_tuning)
        event = MockEvent([38])  # D2 в Drop D строе

        result = map_event(event, mapper)
        assert (6, 0) in result[0]  # открытая 6-я струна

    def test_map_event_empty_event(self):
        """Тест маппинга пустого события"""
        event = MockEvent([])
        result = map_event(event)
        assert result == []

    def test_map_event_with_notes_not_on_guitar(self):
        """Тест маппинга нот, которых нет на гитаре"""
        event = MockEvent([10, 20, 30])
        result = map_event(event)

        for positions in result:
            assert positions == []  # все возвращают пустые списки

# Запуск тестов:
# pytest test_fret_mapper.py -v