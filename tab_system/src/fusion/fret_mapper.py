from config import STANDARD_TUNING

def get_positions(midi_note):
    positions = []

    for string_idx, open_note in enumerate(STANDARD_TUNING):
        fret = midi_note - open_note

        if 0 <= fret <= 24:
            positions.append((string_idx, fret))

    return positions


def map_event(event):
    result = []

    for note in event.notes:
        positions = get_positions(note.midi)
        result.append(positions)

    return result