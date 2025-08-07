# Adversarial Attack Library

A simple library for generating adversarial examples against ResNet50 using PGD attacks.

## Prerequisites
- uv package manager (`pip install uv` or see [installation guide](https://docs.astral.sh/uv/getting-started/installation/))

## Installation

```bash
git clone <repository-url>
cd adv_attack
uv sync
```

## Usage

### Testing the Package
```bash
uv run simple_test.py
```

### Quick run 
```bash
uv run demo.py test_image.jpg 285 adversarial_image.jpg
```

### Interactive Python/IPython
```bash
uv sync
uv pip install -e .
uv run python  # Important: Use UV's Python, NOT system python!
```

Then in Python:
```python
from adv_attack import ResNet50Wrapper, pgd_attack, load_image, tensor_to_pil

# Load model
model = ResNet50Wrapper()

# Load and classify original image
image = load_image("test_image.jpg")
original_class_id, original_class_name = model.classify(image)
print(f"Original: {original_class_name}")

# Generate adversarial example
target_class = 285  # Egyptian cat
adversarial_image = pgd_attack(model, image, target_class)

# Classify adversarial image
adv_class_id, adv_class_name = model.classify(adversarial_image)
print(f"Adversarial: {adv_class_name}")

# Save result
adversarial_pil = tensor_to_pil(adversarial_image)
adversarial_pil.save("adversarial_output.jpg")
```

### Jupyter Notebook
```bash
uv sync
uv add jupyter
uv run jupyter notebook  # Use UV's jupyter
```

### Important Notes
- **Always use `uv run python`** instead of system `python`
- **For Jupyter**: Use `uv run jupyter notebook`
- The package is installed in UV's virtual environment, not your system Python

## Directory Structure
```
adv_attack/
├── pyproject.toml
├── README.md
├── src/
│   └── adv_attack/
│       ├── __init__.py
│       ├── attack.py      # PGD attack implementation
│       ├── models.py      # ResNet50 wrapper
│       └── utils.py       # Image utilities
├── demo.py               # Command-line demo script
├── imagenet_class_index.json  # ImageNet class names
```
