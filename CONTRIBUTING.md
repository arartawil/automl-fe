# Contributing to XAIevo

Thank you for your interest in contributing to XAIevo! We welcome contributions from the community and are grateful for your help in making this project better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Security](#security)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include details about your configuration and environment**

Use this template for bug reports:

```
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.10.0]
- XAIevo version: [e.g. 0.1.0]

**Additional context**
Add any other context about the problem here.
```

### Requesting Features

Feature requests are welcome! Before submitting a feature request:

- **Check if the feature has already been requested**
- **Provide a clear and detailed explanation of the feature**
- **Explain why this feature would be useful**
- **Consider the scope and complexity**

Use this template for feature requests:

```
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Setting up your development environment

1. **Fork the repository** on GitHub
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/your-username/xaievo.git
   cd xaievo
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

5. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov black ruff mypy
   ```

6. **Create a new branch for your feature**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Pull Request Process

1. **Ensure any install or build dependencies are removed** before submitting
2. **Update the README.md** with details of changes if applicable
3. **Add or update tests** for your changes
4. **Ensure the test suite passes**: `pytest tests/`
5. **Run code formatting**: `black src/ tests/`
6. **Run linting**: `ruff check src/ tests/`
7. **Update documentation** if needed

### Pull Request Checklist

Before submitting your pull request, make sure:

- [ ] Code follows the established style guidelines
- [ ] Tests pass (`pytest tests/`)
- [ ] Code is properly formatted (`black src/ tests/`)
- [ ] Linting passes (`ruff check src/ tests/`)
- [ ] Documentation is updated if necessary
- [ ] Commit messages follow the guidelines
- [ ] The PR description clearly describes the changes

## Code Style

We use several tools to maintain code quality:

### Formatting with Black

We use [Black](https://github.com/psf/black) for code formatting:

```bash
# Format all code
black src/ tests/

# Check formatting without making changes
black --check src/ tests/
```

### Linting with Ruff

We use [Ruff](https://github.com/astral-sh/ruff) for fast Python linting:

```bash
# Lint code
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/
```

### Type Checking with MyPy

We encourage type hints and use MyPy for type checking:

```bash
# Type check
mypy src/
```

### Code Style Guidelines

- Use descriptive variable and function names
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use type hints where appropriate
- Follow PEP 8 conventions (enforced by Black and Ruff)

## Testing

We use [pytest](https://docs.pytest.org/) for testing.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=xaievo --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run tests matching a pattern
pytest tests/ -k "test_tracker"
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Test both happy paths and error conditions
- Aim for high test coverage (>80%)
- Place tests in the `tests/` directory

Example test:

```python
import pytest
from xaievo import RunTracker

def test_run_tracker_basic_functionality():
    """Test basic RunTracker operations."""
    tracker = RunTracker(run_id="test_run")
    
    tracker.log_iteration(
        iteration=0,
        best_fitness=10.0,
        best_solution=[1.0, 2.0]
    )
    
    summary = tracker.get_summary()
    assert summary["run_id"] == "test_run"
    assert summary["total_iterations"] == 1
    assert summary["best_fitness"] == 10.0
```

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Examples

```
feat: add feature importance computation

fix(tracker): handle empty snapshots in get_summary()

docs: update README with new installation instructions

test: add tests for RunTracker edge cases
```

## Security

If you discover a security vulnerability, please report it privately by emailing [security@example.com] instead of opening a public issue. We will address security issues promptly.

### Security Best Practices

- Never commit secrets, API keys, or passwords
- Use secure coding practices
- Report vulnerabilities responsibly
- Keep dependencies updated

## Questions?

If you have questions about contributing, please:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/yourusername/xaievo/issues)
3. Open a [discussion](https://github.com/yourusername/xaievo/discussions)
4. Contact the maintainers

## Recognition

Contributors will be recognized in our README and release notes. We appreciate all forms of contribution, from code to documentation to bug reports!

Thank you for contributing to XAIevo! 🎉