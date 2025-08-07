# Adversarial Attack Library

A simple library for generating adversarial examples against ResNet50 using PGD attacks.

## Prerequisites
- uv package manager (<code>pip install uv</code> or see [here](https://docs.astral.sh/uv/getting-started/installation/) for more info)

## Installation

```bash
git clone git@github.com:roshanaz/adv_attack.git
cd adv_attack
uv sync
```

## Usage

```python
uv run demo.py test_image.jpg 285 adversarial_image.jpg
```

## Example of output
```bash
Original prediction: Class 282 (tiger cat)
Target class: 291 (lion)
Adversarial prediction: Class 291 (lion)
Attack succeeded! Model fooled into predicting target class.
Perturbation Metrics:
l_inf norm: 0.007184
L2 norm: 1.943249
Mean absolute change: 0.004423
Max pixel change: 1.8/255 (in original scale)
Perturbation within imperceptible range

Saved adversarial image to adversarial_image.jpg
```

## Directory Structure Created:
```
adv_attack/
├── pyproject.toml
├── README.md
├── src/
│   └── adv_attack/
│       ├── __init__.py
│       ├── attack.py
│       ├── models.py
│       └── utils.py
├── demo.py
└── tests/
    
```
