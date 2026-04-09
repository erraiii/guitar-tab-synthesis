import pytest
from fusion.candidates import generate_visual_candidates


def test_multiple_fingers_same_string():
    fingers = [(2, 3), (2, 5)]
    result = generate_visual_candidates(fingers)

    assert (2, 0) in result
    assert (2, 3) in result
    assert (2, 5) in result


def test_total_length():
    fingers = [(2, 3), (3, 2)]
    result = generate_visual_candidates(fingers)

    # 6 открытых + 2 пальца
    assert len(result) == 8


def test_no_fingers():
    result = generate_visual_candidates([])

    assert len(result) == 6
    assert all(f == 0 for _, f in result)


def test_with_capo():
    result = generate_visual_candidates([(2, 6)], capo=4)

    assert (2, 4) in result
    assert (2, 6) in result


def test_all_strings_present():
    result = generate_visual_candidates([(2, 3)])

    strings = {s for s, _ in result}
    assert strings == {1, 2, 3, 4, 5, 6}


def test_structure():
    result = generate_visual_candidates([(2, 3)])

    for s, f in result:
        assert isinstance(s, int)
        assert isinstance(f, int)