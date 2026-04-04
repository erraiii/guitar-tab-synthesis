from dataclasses import dataclass
from typing import List, Optional
import numpy as np

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



@dataclass
class Detection:
    class_id: int
    class_name: str
    corners: np.ndarray  # (4, 2)

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "class": self.class_name,
            "corners": self.corners.tolist()
        }


@dataclass
class Fret:
    corners: np.ndarray
    index: int | None = None

    @property
    def center(self):
        return self.corners.mean(axis=0)


@dataclass
class Nut:
    corners: np.ndarray

    @property
    def center(self):
        return self.corners.mean(axis=0)


@dataclass
class Capo:
    corners: np.ndarray

    @property
    def center(self):
        return self.corners.mean(axis=0)


@dataclass
class GuitarDetections:
    frets: List[Fret]
    nut: Optional[Nut] = None
    capo: Optional[Capo] = None
    time: float | None = None

    @property
    def num_frets(self) -> int:
        return len(self.frets)