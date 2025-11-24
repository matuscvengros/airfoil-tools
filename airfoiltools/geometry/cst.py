from dataclasses import dataclass, field
import numpy as np
from scipy.special import comb
from typing import Optional, Union

from airfoiltools.geometry.airfoil import Airfoil

@dataclass(slots=True, init=False)
class CST(Airfoil):
    # Arguments for CST airfoil definition
    w_upper: np.ndarray
    w_lower: np.ndarray
    w_le: float
    Delta_TE: float

    @property
    def order(self) -> Union[int, dict[str, int]]:
        """Return the order of the Bernstein polynomials used for upper and lower surfaces."""
        if len(self.w_upper) == len(self.w_lower):
            return len(self.w_upper) - 1
        else:
            return {"upper": len(self.w_upper) - 1, "lower": len(self.w_lower) - 1}


    def __init__(self, w_upper: Union[float, np.ndarray], w_lower: Union[float, np.ndarray], w_le: float, Delta_TE: float, name: str = "Unnamed CST Airfoil", chord: float = 1.0) -> None:
        super(CST, self).__init__(name, chord)

        self.w_upper = w_upper
        self.w_lower = w_lower
        self.w_le = w_le
        self.Delta_TE = Delta_TE

        self.__post_init__()


    def __post_init__(self) -> None:
        # Generate airfoil coordinates
        psi, zeta = self.generate_coordinates(self.w_upper, self.w_lower, self.w_le, self.Delta_TE)

        # Create coordinates in Selig format
        x = self.chord * np.concatenate((psi[::-1], psi[1:]))
        y = self.chord * np.concatenate((zeta["upper"][::-1], zeta["lower"][1:]))
        
        self.psi = psi
        self.upper = zeta["upper"]
        self.lower = zeta["lower"]

        # Create one airfoil coordinate array in SELIG format
        y = self.chord * np.concatenate((self.upper[::-1], self.lower[1:]))
        self.coords = np.column_stack((x, y))

        super(CST, self).__post_init__()


    def generate_coordinates(self, w_upper, w_lower, w_le, Delta_TE, num_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Generate airfoil coordinates using the CST method."""
        # Generate cosine spacing
        beta = np.linspace(0, np.pi, num_points)
        psi = 0.5 * (1.0 - np.cos(beta))
        
        # Define the class function for airfoil shape
        C = np.sqrt(psi) * (1 - psi)
        
        # Define the shape function
        def S(weights, psi):
            n = len(weights) - 1
            shape_fun = np.zeros_like(psi)
            
            for k in range(n + 1):
                K = comb(n, k)
                base = K * (psi**k) * ((1 - psi)**(n - k))

                shape_fun += weights[k] * base
                
            return shape_fun

        S_upper = S(w_upper, psi)
        S_lower = S(w_lower, psi)
        
        # LE correction term
        def le_corr(w_le, weights, psi):
            N = len(weights)

            return w_le * psi * (1 - psi)**(N + 0.5)

        # Calculate upper and lower surface coordinates
        upper = C * S_upper + le_corr(w_le, w_upper, psi) + psi * (Delta_TE / 2.0)
        lower = C * S_lower + le_corr(w_le, w_lower, psi) - psi * (Delta_TE / 2.0)

        zeta = {"upper": upper, "lower": lower}
        
        return psi, zeta
