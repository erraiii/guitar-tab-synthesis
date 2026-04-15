from config import AUDIO_MIN_NOTE_DURATION, AUDIO_DUPLICATE_WINDOW, AUDIO_GROUPING_THRESHOLD
from core.models import AudioNote, AudioEvent


def filter_notes_by_len(notes):
    return [n for n in notes if (n.end - n.start) > AUDIO_MIN_NOTE_DURATION]


def remove_duplicates(notes):
    result = []

    for note in notes:
        duplicate = False

        for r in result:
            if (
                abs(note.start - r.start) < AUDIO_DUPLICATE_WINDOW
                and note.pitch == r.pitch
            ):
                duplicate = True
                break

        if not duplicate:
            result.append(note)

    return result


def group_notes(notes):
    notes = sorted(notes, key=lambda x: x.start)

    events = []
    current = []

    for note in notes:
        if not current:
            current.append(note)
            continue

        if abs(note.start - current[0].start) < AUDIO_GROUPING_THRESHOLD:
            current.append(note)
        else:
            events.append(current)
            current = [note]

    if current:
        events.append(current)

    return events


def build_events(groups):
    events = []

    for group in groups:
        # notes = [n.notes[0] for n in group]

        events.append(
            AudioEvent(
                start=min(n.start for n in group),
                end=max(n.end for n in group),
                notes=group
            )
        )

    return events


def process_notes(notes):
    print(f"  [process_notes] вход: {len(notes)} нот")
    notes = filter_notes_by_len(notes)
    notes = remove_duplicates(notes)
    groups = group_notes(notes)
    events = build_events(groups)
    print(f"  [process_notes] событий: {len(events)}")
    return events