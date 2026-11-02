"""PCNN -- Physics-Constrained Neural Networks: a controlled comparison.

Four ways of putting a conservation law into a network, scored on the same
data, with the same budget, over error / stability / extrapolation / noise.
"""

from .data import make_data
from .models import MODELS, make_model
from .systems import make_system
from .train import TrainConfig, train

__version__ = "1.0.0"
__all__ = ["make_data", "make_model", "make_system", "train", "TrainConfig", "MODELS"]
