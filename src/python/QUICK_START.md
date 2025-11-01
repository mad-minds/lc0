# lc0-keras Quick Start

## Install

```bash
uv add git+https://github.com/LeelaChessZero/lc0#subdirectory=src/python
```

## Use

```python
import lc0keras
import keras

model = keras.models.load_model('network.keras', safe_mode=True)
policy, value, mlh = model.predict(board_input)
```

That's it! 🎉

## Convert LCZero Networks

```bash
lc0 leela2keras --input=net.pb.gz --output=net.keras
```

## Features

✅ Safe mode loading (`safe_mode=True`)  
✅ Zero Lambda layers  
✅ Dynamic batching  
✅ Full serialization support  
✅ Production-ready  

## More Info

- [Package Documentation](README.md)
- [Detailed Usage Guide](../../KERAS_USAGE.md)
- [Package Structure](PACKAGE_INFO.md)

