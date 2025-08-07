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
