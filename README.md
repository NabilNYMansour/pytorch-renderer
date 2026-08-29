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

## Hello Square

```bash
source .venv/bin/activate
python hello-square.py
```

## Hello Circle

```bash
source .venv/bin/activate
python hello-circle.py
```

## Checkerboard

```bash
source .venv/bin/activate
python checkerboard.py
```

## Conway's Game of Life

```bash
source .venv/bin/activate
python conway.py
```

## Desmos

```bash
source .venv/bin/activate
python desmos.py
```

# Purpose

The point of this repo is to learn how pytorch works by wrapping it around a graphics pipeline and using it to render graphics.
