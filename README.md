# Learning How To Swim

A planar 3-link swimmer has to reach a goal navigating circular obstacles.
It has a forward only tail thruster with a lag, two joints, and
anisotropic linear + quadratic drag.

## Setup

Tested on Ubuntu 20.04, Python 3.11, CPU only.

```bash
conda create -n swimmer python=3.11 -y && conda activate swimmer
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Train

```bash
python -m swimmer_nav.train            
```

## Evaluate

```bash
python -m swimmer_nav.evaluate --run runs/ppo
```

To watch the policy:

```bash
python -m swimmer_nav.evaluate --run runs/ppo --view
```

