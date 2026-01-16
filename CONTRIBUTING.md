# Contributing to WLAN Indoor Localization

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

Be respectful, constructive, and professional in all interactions.

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/wlan_localization
cd wlan_localization

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/wlan_localization
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Development Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Download Data

```bash
python scripts/download_data.py
```

### 4. Verify Setup

```bash
# Run tests
pytest

# Check code quality
pre-commit run --all-files
```

---

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-deep-learning-model`
- `bugfix/fix-preprocessing-nan`
- `docs/improve-readme`
- `test/add-integration-tests`

### Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Make Your Changes

1. **Write code** following our style guide
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run tests locally** before committing

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models/test_cascade.py

# Run with coverage
pytest --cov=wlan_localization --cov-report=html

# Run fast tests only (skip slow integration tests)
pytest -m "not slow"
```

### Writing Tests

**Location**: `tests/test_MODULE/test_file.py`

**Structure**:
```python
"""Tests for MODULE functionality."""

import pytest
from wlan_localization.MODULE import ClassName


class TestClassName:
    """Test ClassName class."""

    def test_initialization(self):
        """Test object initialization."""
        obj = ClassName(param=value)
        assert obj.param == value

    def test_method_with_valid_input(self, fixture_name):
        """Test method with valid input."""
        result = obj.method(fixture_name)
        assert result == expected
```

**Fixtures**: Define in `tests/conftest.py`

**Coverage Target**: 80%+ for new code

---

## Code Style

### Python Style Guide

We follow **PEP 8** with modifications:
- **Line length**: 100 characters (not 79)
- **String quotes**: Double quotes preferred
- **Imports**: Organized with isort

### Type Hints

**Required** for all public functions:

```python
from typing import Optional, Tuple
from numpy.typing import NDArray
import numpy as np

def process_data(
    X: NDArray[np.float64],
    threshold: float = 0.5,
    normalize: bool = True
) -> Tuple[NDArray[np.float64], dict]:
    """Process input data.

    Args:
        X: Input feature array
        threshold: Filtering threshold
        normalize: Whether to normalize output

    Returns:
        Tuple of (processed array, metadata dict)
    """
    ...
```

### Docstrings

**Google Style** required:

```python
def function_name(param1: str, param2: int) -> bool:
    """One-line summary.

    Detailed explanation of what the function does,
    if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        RuntimeError: When operation fails

    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    ...
```

### Formatting Tools

**Automatic formatting** with pre-commit:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

**Pre-commit hooks** run automatically on `git commit`

---

## Code Quality Checklist

Before submitting, ensure:

- [ ] All tests pass locally
- [ ] Code coverage ≥80% for new code
- [ ] Type hints added for all functions
- [ ] Docstrings added (Google style)
- [ ] Code formatted with black
- [ ] Imports sorted with isort
- [ ] No flake8 errors
- [ ] mypy passes (or issues documented)
- [ ] Pre-commit hooks pass
- [ ] Documentation updated if needed

---

## Submitting Changes

### 1. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: Add deep learning model for position prediction

- Implement CNN architecture for RSSI pattern recognition
- Add training pipeline with early stopping
- Update documentation with DL approach
- Add unit tests for new model"
```

**Commit Message Format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructure
- `test`: Adding tests
- `chore`: Maintenance

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your branch
4. Fill in PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guide
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
```

### 4. Code Review

- Address reviewer feedback
- Make requested changes
- Push updates to same branch
- PR will update automatically

---

## Development Workflow

### Typical Workflow

```bash
# 1. Sync with upstream
git fetch upstream
git checkout master
git merge upstream/master

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes and test
# ... edit code ...
pytest
pre-commit run --all-files

# 4. Commit and push
git add .
git commit -m "feat: Add my feature"
git push origin feature/my-feature

# 5. Create PR on GitHub

# 6. After PR merged, cleanup
git checkout master
git pull upstream master
git branch -d feature/my-feature
```

---

## Project Structure

Understanding the codebase:

```
wlan_localization/
├── src/wlan_localization/     # Source code
│   ├── data/                  # Data loading/preprocessing
│   ├── models/                # ML models
│   ├── evaluation/            # Metrics and visualization
│   ├── cli/                   # Command-line interfaces
│   └── utils/                 # Utilities
├── tests/                     # Test suite
│   ├── test_data/
│   ├── test_models/
│   └── test_evaluation/
├── configs/                   # YAML configurations
├── notebooks/                 # Jupyter notebooks
├── docs/                      # Documentation
└── scripts/                   # Utility scripts
```

---

## Common Tasks

### Adding a New Model

1. Create `src/wlan_localization/models/my_model.py`
2. Implement with proper type hints and docstrings
3. Add tests in `tests/test_models/test_my_model.py`
4. Update `src/wlan_localization/models/__init__.py`
5. Add example to docs

### Adding a New Metric

1. Create function in `src/wlan_localization/evaluation/metrics.py`
2. Add tests in `tests/test_evaluation/test_metrics.py`
3. Update docs with usage example

### Updating Documentation

1. Edit relevant `.md` file in `docs/`
2. Update docstrings if API changed
3. Update README.md if necessary

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Features**: Open a GitHub Issue with [Feature Request] tag
- **Security**: Email maintainer directly (see README)

---

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing!
