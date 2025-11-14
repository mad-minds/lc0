"""
Custom Keras layers for Leela Chess Zero models.

This module contains custom serializable layers used by the LCZero Keras converter.
Import this module before loading LCZero Keras models to register the custom layers.
"""

from keras import ops
import keras as keras_core
from keras import layers

# Register all custom layers with Keras serialization
@keras_core.saving.register_keras_serializable()
class MatMul(layers.Layer):
    """Matrix multiplication layer that properly serializes."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return ops.matmul(inputs[0], inputs[1])
    
    def compute_output_shape(self, input_shapes):
        # Batch matmul: (..., m, k) x (..., k, n) -> (..., m, n)
        return input_shapes[0][:-1] + (input_shapes[1][-1],)

@keras_core.saving.register_keras_serializable()
class Gather(layers.Layer):
    """Gather layer with axis parameter - backend agnostic."""
    def __init__(self, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        # Backend-agnostic gather implementation
        # Use swapaxes + take + swapaxes for maximum compatibility
        data, indices = inputs[0], inputs[1]
        
        # Convert indices to int32 if needed
        indices = ops.cast(indices, dtype='int32')
        
        # For axis=1 (the common case), use swapaxes approach
        if self.axis == 1:
            # Move axis 1 to position 0
            data_swapped = ops.swapaxes(data, 0, 1)  # (dim1, batch, ...)
            
            # Gather along axis 0 - this preserves the remaining dimensions
            gathered_swapped = ops.take(data_swapped, indices, axis=0)  # (num_indices, batch, ...)
            
            # Swap back: (batch, num_indices, ...)
            gathered = ops.swapaxes(gathered_swapped, 0, 1)
            return gathered
        elif self.axis == 0:
            # For axis=0, ops.take with axis=0 should work directly
            return ops.take(data, indices, axis=0)
        else:
            # For other axes, swap to 0, gather, swap back
            data_swapped = ops.swapaxes(data, 0, self.axis)
            gathered_swapped = ops.take(data_swapped, indices, axis=0)
            return ops.swapaxes(gathered_swapped, 0, self.axis)
    
    def compute_output_shape(self, input_shapes):
        # Gathering along axis, indices shape replaces that dimension
        if self.axis == 1:
            return (input_shapes[0][0], input_shapes[1][0]) + input_shapes[0][2:]
        return input_shapes[0]
    
    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config

@keras_core.saving.register_keras_serializable()
class DynamicTile(layers.Layer):
    """Tile layer that handles dynamic batch size."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # inputs[0] is the reference tensor for batch size
        # inputs[1] is the tensor to tile (must be shape-compatible)
        # For JAX JIT compatibility, we need to avoid ops.tile with dynamic repeats
        # Instead, use ops.repeat along axis 0 and then reshape
        batch_size = ops.shape(inputs[0])[0]
        # Expand to add batch dimension: (1, ...) -> (batch, ...)
        expanded = ops.expand_dims(inputs[1], axis=0)  # (1, ...)
        # Repeat along axis 0 batch_size times
        # Use ops.repeat which is JIT-compatible with dynamic repeats
        tiled = ops.repeat(expanded, batch_size, axis=0)  # (batch, ...)
        return tiled
    
    def compute_output_shape(self, input_shapes):
        # Output shape: (batch, *input_shapes[1])
        return (input_shapes[0][0],) + input_shapes[1]

@keras_core.saving.register_keras_serializable()
class StaticRepeat(layers.Layer):
    """Repeat layer with fixed repetition count."""
    def __init__(self, repeats, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.repeats = repeats
        self.axis = axis
    
    def call(self, inputs):
        expanded = ops.expand_dims(inputs, axis=self.axis)
        return ops.repeat(expanded, self.repeats, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config['repeats'] = self.repeats
        config['axis'] = self.axis
        return config

@keras_core.saving.register_keras_serializable()
class SliceLayer(layers.Layer):
    """Slice layer with start and size parameters."""
    def __init__(self, start_indices, sizes, **kwargs):
        super().__init__(**kwargs)
        self.start_indices = start_indices
        self.sizes = sizes
    
    def call(self, inputs):
        return ops.slice(inputs, self.start_indices, self.sizes)
    
    def get_config(self):
        config = super().get_config()
        config['start_indices'] = self.start_indices
        config['sizes'] = self.sizes
        return config

@keras_core.saving.register_keras_serializable()
class FlattenBatchSpatial(layers.Layer):
    """Flatten batch and spatial dimensions: (batch, 64, C) -> (batch*64, C)."""
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
    
    def call(self, inputs):
        return ops.reshape(inputs, (-1, self.channels))
    
    def compute_output_shape(self, input_shape):
        return (None, self.channels)
    
    def get_config(self):
        config = super().get_config()
        config['channels'] = self.channels
        return config

@keras_core.saving.register_keras_serializable()
class UnflattenBatchSpatial(layers.Layer):
    """Unflatten batch and spatial dimensions: (batch*64, C) -> (batch, 64, C)."""
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
    
    def call(self, inputs):
        return ops.reshape(inputs, (-1, 64, self.channels))
    
    def compute_output_shape(self, input_shape):
        return (None, 64, self.channels)
    
    def get_config(self):
        config = super().get_config()
        config['channels'] = self.channels
        return config

@keras_core.saving.register_keras_serializable()
class DynamicReshape(layers.Layer):
    """Dynamic reshape with -1 for automatic dimension inference."""
    def __init__(self, target_shape, output_shape_tuple, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape  # tuple with -1 for auto dim
        self.output_shape_tuple = output_shape_tuple  # for compute_output_shape
    
    def call(self, inputs):
        return ops.reshape(inputs, self.target_shape)
    
    def compute_output_shape(self, input_shape):
        return self.output_shape_tuple
    
    def get_config(self):
        config = super().get_config()
        config['target_shape'] = self.target_shape
        config['output_shape_tuple'] = self.output_shape_tuple
        return config

