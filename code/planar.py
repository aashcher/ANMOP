"""
    MIT Licence, © Alexey A. Shcherbakov, 2024
"""

import numpy as np
from copy import copy, deepcopy

class Interface:
    def __init__(self, epsilon1: complex, epsilon2: complex) -> None:
        self.eps1 = epsilon1.copy()
        self.eps2 = epsilon2.copy()
        pass

class Layer:
    def __init__(self, k_thickness: float, epsilon: complex) -> None:
        self.kh = k_thickness.copy()
        self.eps = epsilon.copy()
        pass

    def calc_kz(self, kx, epsilon):
        kz = np.sqrt(epsilon - complex(kx*kx))
        if isinstance(kz, np.ndarray):
            kz[np.angle(kz) < -1e-8] = -kz[np.angle(kz) < -1e-8]
        else:
            if np.angle(kz) < -1e-8:
                kz = -kz
        return kz

class PlanarWaveguide:
    def __init__(self, k_thicknesses: np.array[float], epsilons: np.array[complex]) -> None:
        if (k_thicknesses.size() + 2) != epsilons.size():
            raise ValueError("Incompatible size of input arrays in PlanarWaveguide.__init__")
        self.kh = k_thicknesses.copy()
        self.eps = epsilons.copy()
        self.n_layers = k_thicknesses.size()
        pass