from fusion.fret_mapper import get_positions


def test_get_positions():

    positions = get_positions(64)

    assert len(positions) > 0