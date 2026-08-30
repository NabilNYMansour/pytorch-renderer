# Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ROCm

Double check if that is the ROCm version of your AMD GPU. Mine is RX 9060-class

```bash
pip install -r requirements-rocm.txt
```

### CUDA

Otherwise if you have a CUDA GPU

```bash
pip install -r requirements-cuda.txt
```

###

# Usage

```bash
source .venv/bin/activate
```

## Basic

### Hello Square

```bash
python basic/hello-square.py
```

### Hello Circle

```bash
python basic/hello-circle.py
```

### Checkerboard

```bash
python basic/checkerboard.py
```

### Desmos

```bash
python basic/desmos.py
```

### Ray Directions

```bash
python basic/ray-directions.py
```

## Advanced

### Conway's Game of Life

```bash
python advanced/conway.py
```

### Raytrace

```bash
python advanced/raytrace.py
```


# Purpose

The point of this repo is to learn how pytorch works by wrapping it around a graphics pipeline and using it to render graphics.
