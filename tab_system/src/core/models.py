from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Note:
    midi: int
    frequency: float


@dataclass
class AudioNote:
    timestamp: float
    notes: List[Note]


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