from .__about__ import __version__

from ._dispersion import DispersionCurve, PhaseDispersion, GroupDispersion
from ._surf96 import surf96

__all__ = [
    "surf96",
    "DispersionCurve",
    "PhaseDispersion",
    "GroupDispersion",
    "__version__",
]
