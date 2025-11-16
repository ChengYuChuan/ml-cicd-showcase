# Contributing to ML CI/CD Showcase

Thank you for your interest in contributing to this project! This is primarily a showcase project for job applications, but suggestions and improvements are welcome.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists
2. Open a new issue with a clear title and description
3. Include:
   - Your Python version
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Any relevant error messages

### Suggesting Enhancements

For feature suggestions:

1. Open an issue with tag `enhancement`
2. Describe the feature and its benefits
3. Explain how it fits with the project goals

### Code Contributions

#### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-cicd-showcase.git
cd ml-cicd-showcase

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Development Workflow

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following these guidelines:
   - Follow PEP 8 style guide
   - Use type hints where appropriate
   - Add docstrings to functions and classes
   - Keep functions focused and modular

3. **Write tests**:
   - Add unit tests for new functionality
   - Ensure existing tests still pass
   - Aim for >80% code coverage

4. **Run quality checks**:
   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/
   
   # Check linting
   flake8 src/ tests/
   
   # Run tests
   pytest tests/ -v
   
   # Check coverage
   pytest tests/ --cov=src --cov-report=term-missing
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```
   
   Use conventional commit messages:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Docs:` for documentation
   - `Test:` for tests
   - `Refactor:` for code refactoring

6. **Push and create Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```
   
   Then open a PR with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots (if UI changes)

#### Code Style

- **Python**: PEP 8 with 100 character line length
- **Imports**: Sorted with isort (black profile)
- **Formatting**: Black code formatter
- **Type hints**: Preferred but not mandatory
- **Docstrings**: Google style

#### Testing Guidelines

- Place tests in `tests/` directory
- Use pytest fixtures from `conftest.py`
- Mark slow tests with `@pytest.mark.slow`
- Test both success and failure cases
- Mock external dependencies (API calls, file I/O)

Example test structure:
```python
class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality works."""
        # Arrange
        # Act
        # Assert
        pass
    
    def test_edge_case(self):
        """Test edge case handling."""
        pass
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            # Test code
            pass
```

#### Adding New Models

To add a new model type:

1. **Create model class** inheriting from `BaseMLModel`:
   ```python
   from src.models.base_model import BaseMLModel
   
   class MyNewModel(BaseMLModel):
       def train(self, *args, **kwargs):
           # Implementation
           pass
       
       def predict(self, input_data):
           # Implementation
           pass
       
       def evaluate(self, *args, **kwargs):
           # Implementation
           pass
       
       def _save_implementation(self, path):
           # Implementation
           pass
       
       def _load_implementation(self, path):
           # Implementation
           pass
   ```

2. **Add configuration** in `src/config.py`:
   ```python
   @dataclass
   class MyNewModelConfig:
       model_name: str = "MyNewModel"
       # Add your config parameters
   ```

3. **Write tests** in `tests/test_mynewmodel.py`

4. **Update documentation**:
   - Add to README.md
   - Update architecture.md
   - Add usage examples

5. **Verify CI/CD** passes with your changes

## Code Review Process

1. All PRs require review before merging
2. CI/CD pipeline must pass (all tests, linting, etc.)
3. Code coverage should not decrease
4. Changes should be well-documented

## Questions?

Feel free to open an issue with your question or reach out to the maintainer.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
