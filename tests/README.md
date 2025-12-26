# Tests

This directory contains test suites for AutoML-FE.

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=automl_fe --cov-report=html
```

## Test Structure

- `test_core.py` - Tests for core explainability components
- `test_integration.py` - Integration tests
- `test_trackers.py` - RunTracker functionality tests

## Contributing Tests

When adding new features, please include corresponding tests. Aim for >80% code coverage.
