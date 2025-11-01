# lc0-keras

Custom Keras 3 layers for Leela Chess Zero neural networks.

## Overview

This package provides custom serializable Keras layers used by LCZero's Keras converter. These layers enable loading and running LCZero neural networks in Keras with full serialization support and `safe_mode=True`.

## Installation

### From GitHub (via uv)

```bash
uv add git+https://github.com/LeelaChessZero/lc0#subdirectory=src/python
```

### From GitHub (via pip)

```bash
pip install git+https://github.com/LeelaChessZero/lc0#subdirectory=src/python
```

### Local Development

```bash
cd /path/to/lc0/src/python
uv pip install -e .
# or
pip install -e .
```

## Usage

Simply import the package before loading your LCZero Keras model:

```python
import lc0keras  # Registers all custom layers
import keras

# Load model with safe_mode=True (secure, production-ready)
model = keras.models.load_model('your-network.keras', safe_mode=True)

# Run inference
import numpy as np
board_input = np.random.randn(1, 8, 8, 112).astype(np.float32)
policy, value, mlh = model.predict(board_input)
```

## Custom Layers

The package provides the following custom Keras layers:

- **`MatMul`** - Batch matrix multiplication with proper shape inference
- **`Gather`** - Index-based gathering operation (wraps `tf.gather`)
- **`DynamicTile`** - Dynamic batch-aware tiling
- **`StaticRepeat`** - Static repetition along an axis
- **`SliceLayer`** - Tensor slicing operations
- **`FlattenBatchSpatial`** - Reshape `(batch, 64, C)` → `(batch*64, C)`
- **`UnflattenBatchSpatial`** - Reshape `(batch*64, C)` → `(batch, 64, C)`
- **`DynamicReshape`** - Dynamic reshape with `-1` dimension inference

All layers are:
- ✅ Fully serializable with `get_config()` and `from_config()`
- ✅ Registered with `@keras.saving.register_keras_serializable()`
- ✅ Support dynamic batch sizes
- ✅ Compatible with `safe_mode=True` loading

## Converting LCZero Networks to Keras

Use the `lc0` binary with the `leela2keras` command:

```bash
lc0 leela2keras \
    --input=network.pb.gz \
    --output=network.keras \
    --python-output=network_gen.py
```

This will:
1. Convert the LCZero network to Keras format
2. Save weights as `.npy` files
3. Generate a Python script that constructs the model
4. Save the final model as a `.keras` file

## Requirements

- Python 3.8+
- TensorFlow 2.13+ / Keras 3.0+
- NumPy 1.23+

## Model Architecture

LCZero Keras models have:
- **Input**: `(batch, 8, 8, 112)` - Chess board representation
- **Outputs**:
  - Policy: `(batch, 1858)` - Move probabilities
  - Value: `(batch, 3)` - Win/Draw/Loss probabilities
  - MLH: `(batch, 1)` - Moves Left Head prediction

## Features

- 🔒 **Secure**: Loads with `safe_mode=True` (no arbitrary code execution)
- 📦 **Zero Lambda layers**: All operations use native or custom serializable layers
- 🔄 **Dynamic batching**: Works with any batch size
- ⚡ **Fast**: Optimized for inference
- 🎯 **Production-ready**: Full serialization/deserialization support

## License

GPL-3.0-or-later - Same as the main LCZero project.

## Contributing

This package is part of the [Leela Chess Zero](https://github.com/LeelaChessZero/lc0) project. Contributions are welcome!

## Links

- [LCZero Homepage](https://lczero.org/)
- [LCZero GitHub](https://github.com/LeelaChessZero/lc0)
- [LCZero Training](https://github.com/LeelaChessZero/lczero-training)

