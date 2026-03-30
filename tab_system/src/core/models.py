from dataclasses import dataclass
from typing import List, Optional

'''
@dataclass
class Note:
    midi: int
    # frequency: float
'''

@dataclass
class AudioNote:
    start: float
    end: float
    pitch: int


@dataclass
class AudioEvent:
    start: float
    end: float
    notes: List[AudioNote]


@dataclass
class Fingering:
    string: int
    fret: int


@dataclass
class VisualResult:
    fingerings: List[Fingering]
    capo: Optional[int] = None


@dataclass
class TabNote:
    string: int
    fret: int


@dataclass
class Tab:
    notes: List[TabNote]