# Contributing to Psyconstruct

We welcome contributions to Psyconstruct! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/dhanumjayareddybhavanam/psyconstruct.git
cd psyconstruct

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests to verify setup
python -m pytest psyconstruct/tests/
```

## Code Style and Standards

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black psyconstruct/
isort psyconstruct/

# Check linting
flake8 psyconstruct/

# Type checking
mypy psyconstruct/
```

## Contributing Guidelines

### Types of Contributions

We welcome the following types of contributions:

1. **Bug Fixes**: Fix errors in existing functionality
2. **New Features**: Add new features or constructs
3. **Documentation**: Improve documentation and examples
4. **Tests**: Add or improve test coverage
5. **Performance**: Optimize existing code

### Submitting Changes

1. **Fork the Repository**: Create a fork of the main repository
2. **Create a Branch**: Create a feature branch from main
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make Changes**: Implement your changes following our coding standards
4. **Add Tests**: Ensure your changes are well-tested
5. **Run Tests**: Verify all tests pass
   ```bash
   python -m pytest psyconstruct/tests/ -v
   ```
6. **Update Documentation**: Update relevant documentation if needed
7. **Submit Pull Request**: Create a pull request with a clear description

### Code Review Process

All contributions go through code review:

1. Automated checks (tests, linting, type checking)
2. Manual code review by maintainers
3. Discussion and feedback
4. Approval and merge

## Feature Development

### Adding New Constructs

When adding new psychological constructs:

1. **Theoretical Foundation**: Provide clear theoretical basis
2. **Feature Implementation**: Implement required features
3. **Registry Updates**: Update construct registry
4. **Tests**: Add comprehensive tests
5. **Documentation**: Update documentation

### Adding New Features

When adding new features to existing constructs:

1. **Feature Definition**: Define feature clearly
2. **Implementation**: Implement feature extraction
3. **Configuration**: Add configuration options
4. **Testing**: Add unit and integration tests
5. **Documentation**: Document the new feature

## Testing

### Test Coverage

We maintain comprehensive test coverage:

- Unit tests for all functions and methods
- Integration tests for end-to-end workflows
- Edge case testing for error conditions
- Performance tests for scalability

### Running Tests

```bash
# Run all tests
python -m pytest psyconstruct/tests/ -v

# Run specific test modules
python -m pytest psyconstruct/tests/test_behavioral_activation.py -v

# Run with coverage
python -m pytest psyconstruct/tests/ --cov=psyconstruct --cov-report=html
```

### Writing Tests

When writing new tests:

1. **Test Structure**: Use descriptive test names
2. **Test Data**: Use realistic test data
3. **Assertions**: Use clear assertions
4. **Edge Cases**: Test boundary conditions
5. **Documentation**: Document complex test scenarios

## Documentation

### Documentation Types

We maintain several types of documentation:

1. **README.md**: Project overview and quick start
2. **API Documentation**: Detailed API reference
3. **Examples**: Practical usage examples
4. **Scientific Background**: Theoretical foundations
5. **Paper.md**: Academic paper for JOSS submission

### Documentation Standards

- Use clear, concise language
- Include code examples
- Provide theoretical context
- Maintain consistency across documents

## Issue Reporting

### Bug Reports

When reporting bugs:

1. **Use Issue Template**: Follow the bug report template
2. **Provide Details**: Include steps to reproduce
3. **Environment**: Specify Python version and dependencies
4. **Expected vs Actual**: Clearly describe what should happen

### Feature Requests

When requesting features:

1. **Use Case**: Describe the use case
2. **Proposed Solution**: Suggest implementation approach
3. **Alternatives**: Consider alternative solutions
4. **Impact**: Discuss potential impact

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: dhanumjayareddybhavanam@gmail.com

## Release Process

### Version Management

We follow semantic versioning:

- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes (backward compatible)

### Release Checklist

Before releases:

1. **Tests**: All tests must pass
2. **Documentation**: Update documentation
3. **Version**: Update version numbers
4. **Changelog**: Update changelog
5. **Tag**: Create git tag
6. **Release**: Create GitHub release

## Getting Help

If you need help:

1. **Documentation**: Check existing documentation
2. **Issues**: Search existing GitHub issues
3. **Discussions**: Ask in GitHub discussions
4. **Email**: Contact maintainers directly

## Acknowledgments

We thank all contributors who help improve Psyconstruct. Your contributions make this project better for the entire research community.

---

Thank you for contributing to Psyconstruct! 🎉
