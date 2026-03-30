import sys
import os

# чтобы работали импорты
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from audio.postprocessing import (
    filter_notes_by_len,
    remove_duplicates,
    group_notes,
    build_events
)
from core.models import AudioNote


def make_note(start, end, pitch):
    return AudioNote(
        start=start,
        end=end,
        pitch=pitch
    )


def test_filter_notes():
    notes = [
        make_note(0.0, 0.01, 60),  # слишком короткая
        make_note(0.0, 0.1, 62),   # норм
    ]

    result = filter_notes_by_len(notes)

    assert len(result) == 1
    assert result[0].pitch == 62


def test_remove_duplicates():
    notes = [
        make_note(0.0, 0.2, 60),
        make_note(0.01, 0.2, 60),  # дубликат
        make_note(0.5, 0.7, 62),
        make_note(0.0, 0.3, 50)
    ]

    result = remove_duplicates(notes)

    assert len(result) == 3


def test_group_notes():
    notes = [
        make_note(0.0, 0.2, 60),
        make_note(0.01, 0.2, 64),  # рядом, тот же аккорд
        make_note(1.0, 1.2, 67),   # отдельно
    ]

    groups = group_notes(notes)

    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 1


def test_build_events():
    groups = [
        [
            make_note(0.0, 0.2, 60),
            make_note(0.05, 0.3, 64),
        ]
    ]

    events = build_events(groups)

    assert len(events) == 1
    assert len(events[0].notes) == 2
    assert events[0].start == 0.0
    assert events[0].end == 0.3


def test_empty_input():
    from audio.postprocessing import process_notes

    result = process_notes([])

    assert result == []


def test_all_filtered():
    notes = [
        make_note(0.0, 0.01, 60),
        make_note(0.1, 0.11, 62),
    ]

    result = filter_notes_by_len(notes)

    assert result == []


def test_group_threshold_edge():
    notes = [
        make_note(0.0, 0.2, 60),
        make_note(0.049, 0.2, 64),  # должно попасть в ту же группу
        make_note(0.06, 0.2, 67),   # уже новая
    ]

    groups = group_notes(notes)

    assert len(groups) == 2


def test_duplicates_not_removed_if_far():
    notes = [
        make_note(0.0, 0.2, 60),
        make_note(1.0, 1.2, 60),  # далеко, значит не дубликат
    ]

    result = remove_duplicates(notes)

    assert len(result) == 2


def test_full_postprocessing_pipeline():
    from audio.postprocessing import process_notes

    notes = [
        make_note(0.0, 0.01, 60),  # отфильтруется
        make_note(0.0, 0.2, 60),
        make_note(0.01, 0.2, 64),
    ]

    events = process_notes(notes)

    assert len(events) == 1
    assert len(events[0].notes) == 2