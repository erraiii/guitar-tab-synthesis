from dataclasses import dataclass
from typing import List, Optional
import numpy as np


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
    _center_line: tuple | None = None
    string_points: np.ndarray | None = None

    @property
    def center(self):
        return self.corners.mean(axis=0)

    @property
    def center_line(self):
        if self._center_line is None:
            self._center_line = self._compute_center_line()
        return self._center_line

    @center_line.setter
    def center_line(self, value):
        self._center_line = value

    def _compute_center_line(self):
        pts = np.array(self.corners)

        edges = [
            (pts[0], pts[1]),
            (pts[1], pts[2]),
            (pts[2], pts[3]),
            (pts[3], pts[0]),
        ]

        lengths = [np.linalg.norm(a - b) for a, b in edges]
        idx = np.argsort(lengths)[:2]

        midpoints = []
        for i in idx:
            a, b = edges[i]
            midpoints.append((a + b) / 2)

        return midpoints[0], midpoints[1]




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


@dataclass
class FingerPosition:
    string: int
    fret: int

@dataclass
class FingeringFrame:
    timestamp: float
    positions: list[FingerPosition]