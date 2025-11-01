"""
lc0keras - Keras 3 custom layers for Leela Chess Zero models.

This package provides custom serializable Keras layers used by LCZero neural networks.
Import this package before loading LCZero Keras models to register the custom layers.

Example:
    >>> import lc0keras
    >>> import keras
    >>> model = keras.models.load_model('network.keras', safe_mode=True)
"""

__version__ = "0.1.0"
__author__ = "Leela Chess Zero Contributors"
__license__ = "GPL-3.0"

# Import all custom layers to register them
from lc0keras.layers import (
    MatMul,
    Gather,
    DynamicTile,
    StaticRepeat,
    SliceLayer,
    FlattenBatchSpatial,
    UnflattenBatchSpatial,
    DynamicReshape,
)

__all__ = [
    "MatMul",
    "Gather",
    "DynamicTile",
    "StaticRepeat",
    "SliceLayer",
    "FlattenBatchSpatial",
    "UnflattenBatchSpatial",
    "DynamicReshape",
]

