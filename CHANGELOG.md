# Changelog

All notable changes to Psyconstruct will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-02-27

### Added
- Initial public research release
- Complete implementation of 14 digital phenotyping features
- Four psychological constructs (Behavioral Activation, Avoidance, Social Engagement, Routine Stability)
- Registry-based construct aggregation system
- Comprehensive provenance tracking
- Quality assessment metrics
- Extensive test suite (274 tests)
- Complete documentation and examples
- JOSS submission paper

### Features
- **Behavioral Activation Features**:
  - Activity Volume: Rolling sum of accelerometer magnitude
  - Location Diversity: Shannon entropy of GPS locations
  - App Usage Breadth: Entropy of app category usage
  - Activity Timing Variance: Variance in activity timing

- **Avoidance Features**:
  - Home Confinement: Percentage of time at home location
  - Communication Gaps: Maximum duration without outgoing communication
  - Movement Radius: Radius of gyration for spatial movement

- **Social Engagement Features**:
  - Communication Frequency: Count of outgoing communications per day
  - Contact Diversity: Number of unique contacts in rolling windows
  - Initiation Rate: Ratio of outgoing to total communications

- **Routine Stability Features**:
  - Sleep Onset Consistency: Standard deviation of sleep onset times
  - Sleep Duration: Average length of inferred sleep intervals
  - Activity Fragmentation: Entropy of hourly activity distribution
  - Circadian Midpoint: Midpoint between sleep onset and wake times

### Technical Implementation
- Deterministic algorithms for reproducible research
- Multiple normalization methods (z-score, min-max, robust)
- Configurable aggregation parameters
- Comprehensive error handling
- Provenance tracking with computational hashing
- Quality assessment metrics for all features
- Registry-based construct definitions

### Documentation
- Comprehensive README with installation and usage instructions
- API reference documentation
- Practical examples for all features
- Scientific background documentation
- JOSS submission paper
- Contributing guidelines

### Testing
- 274 comprehensive unit and integration tests
- Edge case testing for all functions
- Performance testing with large datasets
- Quality assurance validation
- Test coverage >85%

### Dependencies
- Python 3.8+ compatibility
- Core dependencies: pandas, numpy, scipy, scikit-learn
- Geographic dependencies: geopy, haversine
- Validation dependencies: pydantic, jsonschema
- Development dependencies: pytest, black, isort, flake8

## [Unreleased]

### Planned
- Multi-participant reliability estimation
- Advanced statistical validation frameworks
- Cross-dataset harmonization standards
- Extended construct library
- Machine learning integration
- Real-time processing capabilities

---

## Version History

- **v1.0.1** (2026-02-27): Initial public research release
- **v1.0.0** (Development): Internal development and testing

## Migration Guide

### From v1.0.0 to v1.0.1

No breaking changes. This is the initial public release.

## Support

For questions about upgrading or using Psyconstruct:

- Documentation: https://github.com/dhanumjayareddybhavanam/psyconstruct/docs
- Issues: https://github.com/dhanumjayareddybhavanam/psyconstruct/issues
- Email: dhanumjayareddybhavanam@gmail.com
