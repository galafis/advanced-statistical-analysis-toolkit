# Contributing to Advanced Statistical Analysis Toolkit

Thank you for your interest in contributing to the Advanced Statistical Analysis Toolkit! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

- Report bugs and issues
- Suggest new features or enhancements
- Improve documentation
- Add new statistical methods
- Create examples and tutorials
- Fix bugs
- Improve code quality

## 🚀 Getting Started

### 1. Fork the Repository

Fork the repository to your own GitHub account and clone it locally:

```bash
git clone https://github.com/YOUR_USERNAME/advanced-statistical-analysis-toolkit.git
cd advanced-statistical-analysis-toolkit
```

### 2. Set Up Development Environment

Install required dependencies:

```bash
# Python dependencies
pip install -r requirements.txt

# R packages
Rscript install_r_packages.R
```

### 3. Create a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

## 📝 Code Style Guidelines

### Python

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular
- Use meaningful variable names

Example:

```python
def calculate_statistic(
    data: np.ndarray,
    method: str = 'mean'
) -> float:
    """
    Calculate a statistic from data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str
        Statistic to calculate
        
    Returns
    -------
    result : float
        Calculated statistic
    """
    if method == 'mean':
        return np.mean(data)
    # ... implementation
```

### R

- Follow tidyverse style guide
- Use roxygen2 documentation format
- Keep functions modular and well-documented
- Use meaningful variable names

Example:

```r
#' Calculate descriptive statistics
#'
#' @param data Numeric vector
#' @param na_rm Logical, remove NA values
#' @return List with statistics
calculate_statistics <- function(data, na_rm = TRUE) {
  list(
    mean = mean(data, na.rm = na_rm),
    sd = sd(data, na.rm = na_rm)
  )
}
```

## 🧪 Testing

All contributions should include appropriate tests.

### Running Tests

```bash
# Run all tests
./run_all_tests.sh

# Run Python tests only
pytest tests/test_python_integration.py -v

# Run R tests only
cd tests && Rscript test_r_functions.R
```

### Writing Tests

- Test both expected behavior and edge cases
- Use descriptive test names
- Include tests for error handling

## 📚 Documentation

- Update README.md if adding new features
- Add docstrings to all new functions
- Create examples for new functionality
- Update type hints and parameter descriptions

## 🔄 Submission Process

### 1. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: description of what you added"
```

Good commit message examples:
- `Add: Bootstrap confidence interval function`
- `Fix: Handle edge case in normality test`
- `Docs: Update PCA example with visualization`
- `Test: Add tests for chi-square function`

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create a Pull Request

- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your branch
- Fill out the PR template with:
  - Description of changes
  - Related issues (if any)
  - Testing performed
  - Screenshots (if applicable)

## ✅ Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Docstrings added/updated
- [ ] No breaking changes (or clearly documented)
- [ ] Examples work correctly

## 🐛 Reporting Bugs

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, R version)
- Minimal code example
- Error messages/stack traces

## 💡 Feature Requests

When suggesting features:

- Describe the feature clearly
- Explain the use case
- Provide examples if possible
- Consider backward compatibility

## 📧 Questions?

If you have questions about contributing:

- Open an issue with the "question" label
- Check existing issues and discussions
- Review the documentation

## 🙏 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Give constructive feedback
- Focus on the best outcome for the project

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Advanced Statistical Analysis Toolkit! 🎉
