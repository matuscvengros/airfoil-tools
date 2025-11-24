from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from shapely.geometry import Polygon


@dataclass(slots=True, init=False)
class Airfoil:
    name: Optional[str] = None
    chord: Optional[float] = None

    psi: Optional[np.ndarray] = field(default=None, repr=False)
    upper: Optional[np.ndarray] = field(default=None, repr=False)
    lower: Optional[np.ndarray] = field(default=None, repr=False)
    coords: Optional[np.ndarray] = field(default=None, repr=False)
    polygon: Optional[Polygon] = field(default=None, repr=False)


    def __init__(self, name: str = "Unnamed Airfoil", chord: float = 1.0) -> None:
        self.name = name
        self.chord = chord


    def __post_init__(self) -> None:
        try:
            self.polygon = Polygon(self.coords)
        except (AttributeError, TypeError):
            self.polygon = None


    def area(self) -> float:
        """Calculate the area enclosed by the airfoil using the shoelace formula."""

        return self.polygon.area