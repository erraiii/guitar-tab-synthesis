from core.models import AudioNote, AudioEvent #, Note


def filter_notes_by_len(notes):
    return [n for n in notes if (n.end - n.start) > 0.05] # короткие убрать


def remove_duplicates(notes):
    result = []

    for note in notes:
        duplicate = False

        for r in result:
            if (
                abs(note.start - r.start) < 0.02
                and note.pitch == r.pitch # note.notes[0].midi == r.notes[0].midi
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

    threshold = 0.05

    for note in notes:
        if not current:
            current.append(note)
            continue

        if abs(note.start - current[0].start) < threshold:
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
                notes=group # notes
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