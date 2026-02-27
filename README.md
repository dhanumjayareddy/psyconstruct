# Psyconstruct: A Reproducible Framework for Theory-Informed Digital Phenotyping Research

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/dhanumjayareddybhavanam/psyconstruct/docs)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18792047-blue.svg)](https://doi.org/10.5281/zenodo.18792047)

## 🎯 Statement of Need

Digital phenotyping research faces a critical reproducibility gap: while frameworks like Beiwe, RADAR-base, and mindLAMP provide data collection infrastructure, they lack standardized, theoretically-grounded methods for transforming raw sensor data into psychological construct scores. Researchers must implement custom feature extraction and aggregation pipelines, leading to inconsistent methodologies and irreproducible results across studies.

Psyconstruct addresses this gap by providing a **reproducible, open-source framework** that provides an explicit computational operationalization of selected psychological theories into standardized digital behavioral indicators. Unlike existing platforms that focus on data collection, Psyconstruct provides the **methodological bridge** between raw smartphone sensor data and theoretically meaningful construct scores, enabling consistent digital phenotyping research across laboratories.

> Psyconstruct is intended for research use and does not provide clinical decision support.

---


### Comparison with Existing Software

Psyconstruct differs from existing digital phenotyping platforms in several key ways:

* **Data Processing Focus**: Unlike Beiwe, RADAR-base, and mindLAMP which primarily focus on data collection, Psyconstruct provides standardized methods for transforming raw sensor data into theoretically-grounded construct scores.

* **Theory-Mapped Registry**: Provides an explicit computational operationalization of psychological theories through a structured registry system that maps features to constructs.

* **Reproducibility Guarantees**: Explicit deterministic algorithms with provenance tracking and version-locked methods ensure reproducible research across laboratories.

* **Separation of Concerns**: Clear architectural separation between feature extraction, construct aggregation, and quality assessment enables modular research workflows.

* **Research-Only Design**: Intentionally designed as research methodology infrastructure rather than clinical deployment tools.

---

## 🎯 Overview

Psyconstruct implements a complete digital phenotyping pipeline that extracts meaningful behavioral patterns from smartphone sensor data and aggregates them into theoretically grounded psychological constructs. The system is designed for researchers working with digital behavioral indicators, providing standardized methods for reproducible digital phenotyping studies.

> In v1.0.1, reflective and formative distinctions are documented in the registry but operationalized using linear weighted aggregation.

### Key Features

- **🧠 Four Theoretically Grounded Psychological Constructs**
  - Behavioral Activation (BA) - 4 features
  - Avoidance (AV) - 3 features  
  - Social Engagement (SE) - 3 features
  - Routine Stability (RS) - 4 features

- **📊 14 Digital Behavioral Indicators**
  - Activity volume, location diversity, app usage breadth, timing variance
  - Home confinement, communication gaps, movement radius
  - Communication frequency, contact diversity, initiation rate
  - Sleep onset consistency, sleep duration, activity fragmentation, circadian midpoint

- **🔬 Research-Grade Implementation**
  - Comprehensive provenance tracking
  - Quality assessment metrics
  - Configurable parameters
  - Robust error handling
  - Extensive unit tests
  - Interpretive summaries based on theoretical directionality

### Validation Status

**Feature validation varies by construct:**
- **Theoretical**: Features based on established psychological theory
- **Experimental**: Features with preliminary empirical support
- **Placeholder**: Features requiring further validation

**Note:** This is a research methodology framework. All constructs require empirical validation in specific research contexts before interpretation.

## 🎯 Scope

Psyconstruct provides:
- Feature extraction from smartphone sensor data
- Registry-based construct aggregation
- Deterministic normalization and weighting

Psyconstruct does NOT provide:
- Clinical diagnosis
- Latent variable modeling
- Machine learning pipelines
- Causal inference tools

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dhanumjayareddybhavanam/psyconstruct.git
cd psyconstruct

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from psyconstruct.features import (
    BehavioralActivationFeatures,
    AvoidanceFeatures,
    SocialEngagementFeatures,
    RoutineStabilityFeatures
)
from psyconstruct.constructs import ConstructAggregator

# Initialize feature extractors
ba_features = BehavioralActivationFeatures()
avoidance_features = AvoidanceFeatures()
se_features = SocialEngagementFeatures()
rs_features = RoutineStabilityFeatures()

# Extract features from your data
ba_results = ba_features.activity_volume(gps_data)
avoidance_results = avoidance_features.home_confinement(gps_data)
se_results = se_features.communication_frequency(communication_data)
rs_results = rs_features.sleep_onset_consistency(screen_data)

# Aggregate into construct scores
aggregator = ConstructAggregator()
all_features = {**ba_results, **avoidance_results, **se_results, **rs_results}
construct_scores = aggregator.aggregate_all_constructs(all_features)

print(f"Behavioral Activation: {construct_scores['behavioral_activation'].normalized_score:.2f}")
print(f"Social Engagement: {construct_scores['social_engagement'].normalized_score:.2f}")
```

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Psychological Constructs](#-psychological-constructs)
- [Feature Extraction](#-feature-extraction)
- [Data Requirements](#-data-requirements)
- [API Reference](#-api-reference)
- [Examples](#-examples)
- [Quality Assurance](#-quality-assurance)
- [Clinical Applications](#-clinical-applications)
- [Research Applications](#-research-applications)
- [Contributing](#-contributing)
- [License](#-license)

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│  Feature         │───▶│  Construct      │
│                 │    │  Extraction      │    │  Aggregation    │
│ • GPS           │    │                  │    │                 │
│ • Accelerometer │    │ • BA Features    │    │ • Normalization │
│ • Communication │    │ • AV Features    │    │ • Weighting     │
│ • Screen State  │    │ • SE Features    │    │ • Quality       │
│ • App Usage     │    │ • RS Features    │    │ • Scoring       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Module Structure

```
psyconstruct/
├── features/                 # Feature extraction modules
│   ├── behavioral_activation.py
│   ├── avoidance.py
│   ├── social_engagement.py
│   ├── routine_stability.py
│   └── __init__.py
├── constructs/               # Construct aggregation
│   ├── aggregator.py
│   ├── registry.json
│   └── __init__.py
├── tests/                    # Comprehensive test suite
│   ├── test_behavioral_activation.py
│   ├── test_avoidance.py
│   ├── test_social_engagement.py
│   ├── test_routine_stability.py
│   └── test_aggregator.py
├── examples/                 # Usage examples
│   ├── behavioral_activation_example.py
│   ├── avoidance_features_example.py
│   ├── social_engagement_example.py
│   ├── routine_stability_example.py
│   └── construct_aggregation_example.py
├── utils/                    # Utility functions
│   └── provenance.py
└── README.md
```

## 🧠 Psychological Constructs

### Behavioral Activation (BA)

**Theoretical Foundation**: Reflects the tendency to engage in goal-directed activities and approach behaviors. Low levels are associated with depression and motivational deficits.

**Features**:
- **Activity Volume**: Rolling sum of accelerometer magnitude over 24-hour windows
- **Location Diversity**: Shannon entropy of clustered GPS locations per week
- **App Usage Breadth**: Entropy of app category usage patterns
- **Activity Timing Variance**: Variance in activity timing across days

**Interpretive Notes (Theoretical Directionality)**:
- High scores: Active, engaged, goal-directed behavior
- Low scores: Withdrawal, anhedonia, reduced motivation

### Avoidance (AV)

**Theoretical Foundation**: Measures behavioral avoidance patterns including social withdrawal and spatial constriction. High levels indicate maladaptive avoidance behaviors.

**Features**:
- **Home Confinement**: Percentage of GPS points within identified home cluster
- **Communication Gaps**: Maximum duration without outgoing communication per day
- **Movement Radius**: Radius of gyration representing spatial movement extent

**Interpretive Notes (Theoretical Directionality)**:
- High scores: Withdrawal, isolation, avoidance behaviors
- Low scores: Engagement, exploration, approach behaviors

### Social Engagement (SE)

**Theoretical Foundation**: Captures the quantity and quality of social interactions and communication patterns. Essential for understanding social functioning and support.

**Features**:
- **Communication Frequency**: Count of outgoing communications per day
- **Contact Diversity**: Number of unique contacts in rolling 7-day windows
- **Initiation Rate**: Ratio of outgoing to total communications

**Interpretive Notes (Theoretical Directionality)**:
- High scores: Socially active, diverse network, proactive engagement
- Low scores: Social withdrawal, limited network, passive communication

### Routine Stability (RS)

**Theoretical Foundation**: Reflects the consistency and regularity of daily behavioral patterns, particularly sleep-wake cycles and activity rhythms. Important for circadian health and mental stability.

**Features**:
- **Sleep Onset Consistency**: Standard deviation of sleep onset times across days
- **Sleep Duration**: Average length of inferred sleep intervals
- **Activity Fragmentation**: Entropy of hourly activity distribution
- **Circadian Midpoint**: Midpoint between sleep onset and wake times

**Interpretive Notes (Theoretical Directionality)**:
- High scores: Consistent routines, stable circadian rhythms
- Low scores: Irregular patterns, circadian disruption, chaos

## 🔧 Feature Extraction

### Data Input Formats

#### GPS Data
```python
gps_data = {
    'timestamp': [datetime(2026, 2, 21, 10, 0, 0), ...],
    'latitude': [40.7128, 40.7130, ...],
    'longitude': [-74.0060, -74.0062, ...],
    'accuracy': [5.0, 4.8, ...]  # Optional
}
```

#### Accelerometer Data
```python
accelerometer_data = {
    'timestamp': [datetime(2026, 2, 21, 10, 0, 0), ...],
    'x': [0.1, 0.2, ...],
    'y': [0.3, 0.1, ...],
    'z': [9.8, 9.7, ...]
}
```

#### Communication Data
```python
communication_data = {
    'timestamp': [datetime(2026, 2, 21, 10, 0, 0), ...],
    'direction': ['outgoing', 'incoming', ...],
    'contact': ['friend_1', 'family_1', ...],
    'type': ['call', 'text', ...]  # Optional
}
```

#### Screen State Data
```python
screen_data = {
    'timestamp': [datetime(2026, 2, 21, 10, 0, 0), ...],
    'screen_state': [1, 0, 1, ...]  # 1=on, 0=off
}
```

#### App Usage Data
```python
app_usage_data = {
    'timestamp': [datetime(2026, 2, 21, 10, 0, 0), ...],
    'app_name': ['instagram', 'whatsapp', ...],
    'category': ['social', 'communication', ...],
    'duration_seconds': [300, 120, ...]
}
```

### Feature Extraction Examples

#### Behavioral Activation
```python
from psyconstruct.features import BehavioralActivationFeatures, ActivityVolumeConfig

# Configure for research-grade analysis
config = ActivityVolumeConfig(
    analysis_window_days=7,
    min_data_coverage=0.8,
    outlier_detection=True
)

features = BehavioralActivationFeatures(activity_volume_config=config)

# Extract activity volume
result = features.activity_volume(accelerometer_data)

print(f"Weekly activity count: {result['weekly_activity_count']}")
print(f"Quality score: {result['quality_metrics']['overall_quality']}")
```

#### Avoidance
```python
from psyconstruct.features import AvoidanceFeatures, HomeConfinementConfig

config = HomeConfinementConfig(
    home_radius_meters=100,
    nighttime_hours=[22, 23, 0, 1, 2, 3, 4, 5],
    min_gps_points=50
)

features = AvoidanceFeatures(home_confinement_config=config)

# Extract home confinement
result = features.home_confinement(gps_data)

print(f"Home confinement: {result['home_confinement_percentage']:.1f}%")
print(f"Home location: {result['home_location']}")
```

#### Social Engagement
```python
from psyconstruct.features import SocialEngagementFeatures

features = SocialEngagementFeatures()

# Extract communication frequency
result = features.communication_frequency(communication_data)

print(f"Weekly outgoing: {result['weekly_outgoing_count']}")
print(f"Daily frequency: {result['communication_frequency']['mean_daily_frequency']}")
```

#### Routine Stability
```python
from psyconstruct.features import RoutineStabilityFeatures

features = RoutineStabilityFeatures()

# Extract sleep onset consistency
result = features.sleep_onset_consistency(screen_data)

print(f"Sleep onset SD: {result['sleep_onset_sd_hours']:.2f} hours")
print(f"Mean onset: {result['sleep_onset_consistency']['mean_sleep_onset_hour']:.1f}:00")
```

## 📊 Construct Aggregation

### Normalization Methods

```python
from psyconstruct.constructs import AggregationConfig, ConstructAggregator

# Z-score normalization (default)
config = AggregationConfig(
    normalization_method="zscore",
    within_participant=True,
    aggregation_method="weighted_mean"
)

# Min-max normalization
config = AggregationConfig(
    normalization_method="minmax",
    reference_population="clinical_sample"
)

# Robust normalization (median + MAD)
config = AggregationConfig(
    normalization_method="robust",
    handle_missing="median_impute"
)
```

### Aggregation Examples

#### Single Construct
```python
aggregator = ConstructAggregator(config=config)

# Aggregate behavioral activation
score = aggregator.aggregate_construct(
    "behavioral_activation",
    feature_results,
    participant_id="participant_001",
    reference_data=reference_population
)

print(f"BA Score: {score.normalized_score:.2f}")
print(f"Interpretation: {score.interpretation}")
print(f"Dispersion Interval: {score.dispersion_interval}")  # Note: Represents dispersion, not inferential CI
```

#### All Constructs
```python
# Aggregate all available constructs
construct_scores = aggregator.aggregate_all_constructs(
    feature_results,
    participant_id="participant_001"
)

for construct_name, score in construct_scores.items():
    print(f"{construct_name}: {score.normalized_score:.2f}")
```

#### Export Results
```python
# Export as JSON
aggregator.export_scores(construct_scores, "scores.json", format="json")

# Export as CSV
aggregator.export_scores(construct_scores, "scores.csv", format="csv")
```

## 📋 Data Requirements

### Minimum Data Requirements

| Construct | Features | Minimum Duration | Data Types Required |
|-----------|----------|------------------|-------------------|
| Behavioral Activation | 4 | 7 days | GPS, Accelerometer, App Usage |
| Avoidance | 3 | 7 days | GPS, Communication |
| Social Engagement | 3 | 7 days | Communication |
| Routine Stability | 4 | 14 days | Screen State, Activity |

### Quality Thresholds

- **GPS Data**: ≥70% coverage, ≤50m accuracy
- **Accelerometer**: ≥80% coverage, regular sampling
- **Communication**: ≥5 communications/day
- **Screen State**: ≥70% coverage, clear on/off patterns
- **App Usage**: ≥10 apps/day, category information

### Temporal Resolution

- **GPS**: 1-5 minute intervals
- **Accelerometer**: 1 minute intervals (recommended)
- **Communication**: Event-based (no resampling)
- **Screen State**: 1 minute intervals
- **App Usage**: Event-based with duration tracking

## 📚 API Reference

### Feature Classes

#### BehavioralActivationFeatures
```python
class BehavioralActivationFeatures:
    def __init__(self, 
                 activity_volume_config: Optional[ActivityVolumeConfig] = None,
                 location_diversity_config: Optional[LocationDiversityConfig] = None,
                 app_usage_breadth_config: Optional[AppUsageBreadthConfig] = None,
                 timing_config: Optional[ActivityTimingVarianceConfig] = None):
    
    def activity_volume(self, accelerometer_data: Dict[str, Any]) -> Dict[str, Any]
    def location_diversity(self, gps_data: Dict[str, Any]) -> Dict[str, Any]
    def app_usage_breadth(self, app_usage_data: Dict[str, Any]) -> Dict[str, Any]
    def activity_timing_variance(self, accelerometer_data: Dict[str, Any]) -> Dict[str, Any]
```

#### AvoidanceFeatures
```python
class AvoidanceFeatures:
    def __init__(self,
                 home_confinement_config: Optional[HomeConfinementConfig] = None,
                 communication_gaps_config: Optional[CommunicationGapsConfig] = None,
                 movement_config: Optional[MovementRadiusConfig] = None):
    
    def home_confinement(self, gps_data: Dict[str, Any]) -> Dict[str, Any]
    def communication_gaps(self, communication_data: Dict[str, Any]) -> Dict[str, Any]
    def movement_radius(self, gps_data: Dict[str, Any]) -> Dict[str, Any]
```

#### SocialEngagementFeatures
```python
class SocialEngagementFeatures:
    def __init__(self,
                 freq_config: Optional[CommunicationFrequencyConfig] = None,
                 diversity_config: Optional[ContactDiversityConfig] = None,
                 initiation_config: Optional[InitiationRateConfig] = None):
    
    def communication_frequency(self, communication_data: Dict[str, Any]) -> Dict[str, Any]
    def contact_diversity(self, communication_data: Dict[str, Any]) -> Dict[str, Any]
    def initiation_rate(self, communication_data: Dict[str, Any]) -> Dict[str, Any]
```

#### RoutineStabilityFeatures
```python
class RoutineStabilityFeatures:
    def __init__(self,
                 sleep_onset_config: Optional[SleepOnsetConfig] = None,
                 sleep_duration_config: Optional[SleepDurationConfig] = None,
                 fragmentation_config: Optional[ActivityFragmentationConfig] = None,
                 circadian_config: Optional[CircadianMidpointConfig] = None):
    
    def sleep_onset_consistency(self, screen_data: Dict[str, Any]) -> Dict[str, Any]
    def sleep_duration(self, screen_data: Dict[str, Any]) -> Dict[str, Any]
    def activity_fragmentation(self, activity_data: Dict[str, Any]) -> Dict[str, Any]
    def circadian_midpoint(self, screen_data: Dict[str, Any]) -> Dict[str, Any]
```

### ConstructAggregator

```python
class ConstructAggregator:
    def __init__(self, 
                 config: Optional[AggregationConfig] = None,
                 construct_registry_path: Optional[str] = None):
    
    def aggregate_construct(self, 
                           construct_name: str,
                           feature_results: Dict[str, Any],
                           participant_id: Optional[str] = None,
                           reference_data: Optional[Dict[str, List[float]]] = None) -> ConstructScore
    
    def aggregate_all_constructs(self,
                                feature_results: Dict[str, Any],
                                participant_id: Optional[str] = None,
                                reference_data: Optional[Dict[str, Dict[str, List[float]]]] = None) -> Dict[str, ConstructScore]
    
    def export_scores(self,
                     construct_scores: Dict[str, ConstructScore],
                     output_path: str,
                     format: str = "json") -> None
```

### Configuration Classes

#### AggregationConfig
```python
@dataclass
class AggregationConfig:
    normalization_method: str = "zscore"  # "zscore", "minmax", "robust", "none"
    within_participant: bool = True
    aggregation_method: str = "weighted_mean"  # "weighted_mean", "unweighted_mean", "median"
    handle_missing: str = "exclude"  # "exclude", "mean_impute", "median_impute"
    min_features_required: int = 2
    min_quality_threshold: float = 0.5
    include_feature_scores: bool = True
    include_quality_metrics: bool = True
    include_normalization_params: bool = True
```

## 💡 Examples

### Complete Analysis Pipeline

```python
from psyconstruct.features import *
from psyconstruct.constructs import ConstructAggregator

def analyze_participant(gps_data, accelerometer_data, communication_data, 
                       screen_data, app_usage_data, participant_id):
    """Complete analysis pipeline for a single participant."""
    
    # Initialize feature extractors
    ba_features = BehavioralActivationFeatures()
    avoidance_features = AvoidanceFeatures()
    se_features = SocialEngagementFeatures()
    rs_features = RoutineStabilityFeatures()
    
    # Extract all features
    feature_results = {}
    feature_results['activity_volume'] = ba_features.activity_volume(accelerometer_data)
    feature_results['home_confinement'] = avoidance_features.home_confinement(gps_data)
    feature_results['communication_frequency'] = se_features.communication_frequency(communication_data)
    feature_results['sleep_onset_consistency'] = rs_features.sleep_onset_consistency(screen_data)
    
    # Aggregate into construct scores
    aggregator = ConstructAggregator()
    construct_scores = aggregator.aggregate_all_constructs(
        feature_results, participant_id=participant_id
    )
    
    return construct_scores, feature_results

# Display results
construct_scores, feature_results = analyze_participant(
    gps_data, accelerometer_data, communication_data, 
    screen_data, app_usage_data, "participant_001"
)

print("Construct Scores:")
for construct, score in construct_scores.items():
    print(f"  {construct}: {score.normalized_score:.2f} - {score.interpretation}")
```

### Longitudinal Monitoring

```python
import pandas as pd

def longitudinal_analysis(participant_data_history):
    """Analyze changes in construct scores over time."""
    
    results = []
    for timestamp, data in participant_data_history.items():
        construct_scores, _ = analyze_participant(**data, participant_id=f"participant_{timestamp}")
        results.append({
            'timestamp': timestamp,
            'behavioral_activation': construct_scores['behavioral_activation'].normalized_score,
            'avoidance': construct_scores['avoidance'].normalized_score,
            'social_engagement': construct_scores['social_engagement'].normalized_score,
            'routine_stability': construct_scores['routine_stability'].normalized_score
        })
    
    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp')
```

## 📊 Statistical Assumptions & Limitations

### Key Statistical Assumptions

The aggregation methods in Psyconstruct make the following statistical assumptions:

- **Linear Additivity**: Features contribute linearly to construct scores
- **Feature Independence**: No adjustment for multicollinearity between features
- **No Latent Structure**: Constructs are treated as weighted composites, not latent variables
- **Equal Weighting**: Current weights are theoretical placeholders, not empirically derived

### Known Limitations

- **Single-Participant Reliability**: Reliability estimation requires multi-participant data for item covariances
- **Dispersion vs Confidence Intervals**: Intervals represent dispersion, not statistical confidence intervals
- **Measurement Models**: In v1.0, reflective vs formative distinctions are documented but not yet statistically modeled - all constructs use linear weighted aggregation
- **Validation Status**: Features have varying validation levels (theoretical vs experimental)
- **Feature Weights**: Current weights are theory-driven placeholders and not empirically optimized

### Future Improvements

- Multi-participant reliability estimation
- Multicollinearity assessment and adjustment
- True latent variable modeling
- Empirically-derived feature weights

### Reproducibility Guarantee

Psyconstruct is **deterministic** given identical input data and configuration. The framework guarantees:

- **Identical inputs → identical outputs**
- **Version-locked algorithms** with documented changes
- **Algorithm version stored in provenance metadata**
- **Provenance tracking** for full audit trails
- **No stochastic processes** in core calculations
- **Registry version coupling** for construct definitions

This ensures reproducible research results across computing environments and time periods.

---

## 🔒 Quality Assurance

### Data Quality Metrics

Each feature extraction includes comprehensive quality assessment:

```python
quality_metrics = {
    'coverage_ratio': float,      # Data completeness
    'sampling_rate': float,       # Temporal consistency
    'accuracy_score': float,      # Measurement precision
    'outlier_ratio': float,       # Data cleanliness
    'overall_quality': float      # Composite quality score
}
```

### Provenance Tracking

All operations include complete provenance tracking:

```python
provenance_record = {
    'operation_id': str,
    'operation_type': str,
    'timestamp': datetime,
    'input_parameters': dict,
    'output_summary': dict,
    'duration_seconds': float,
    'algorithm_version': str,
    'data_quality_metrics': dict
}
```

### Validation Results

The system has been validated with:

- **Unit Test Coverage**: Comprehensive test suite with edge case coverage
- **Integration Tests**: End-to-end pipeline validation
- **Preliminary Validation**: Internal testing with simulated datasets. No published empirical validation study accompanies v1.0.
- **Performance Testing**: Scalability to large datasets
- **Quality Assurance**: Automated quality checks

## 🏥 Research Applications

### Population Studies

```python
def population_analysis(participant_data):
    """Analyze digital phenotyping patterns across populations."""
    
    all_scores = []
    
    for participant_id, data in participant_data.items():
        construct_scores, _ = analyze_participant(**data, participant_id=participant_id)
        
        all_scores.append({
            'participant_id': participant_id,
            'ba_score': construct_scores['behavioral_activation'].normalized_score,
            'avoidance_score': construct_scores['avoidance'].normalized_score,
            'se_score': construct_scores['social_engagement'].normalized_score,
            'rs_score': construct_scores['routine_stability'].normalized_score
        })
    
    df = pd.DataFrame(all_scores)
    
    # Population statistics
    population_stats = {
        'mean_scores': df.mean().to_dict(),
        'std_scores': df.std().to_dict(),
        'correlation_matrix': df.corr().to_dict(),
        'sample_size': len(df)
    }
    
    return population_stats, df
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest psyconstruct/tests/ -v

# Run specific test modules
python -m pytest psyconstruct/tests/test_behavioral_activation.py -v
python -m pytest psyconstruct/tests/test_aggregator.py -v

# Run with coverage
python -m pytest psyconstruct/tests/ --cov=psyconstruct --cov-report=html
```

### Test Coverage

The test suite includes:

- **Unit Tests**: Individual function and method testing
- **Integration Tests**: End-to-end pipeline testing
- **Quality Tests**: Data quality and validation testing
- **Performance Tests**: Scalability and efficiency testing

Current coverage: **Comprehensive unit and integration test suite**

### Example Test

```python
def test_behavioral_activation_activity_volume():
    """Test activity volume feature extraction."""
    
    config = ActivityVolumeConfig(analysis_window_days=7)
    features = BehavioralActivationFeatures(activity_volume_config=config)
    
    # Create test data
    accelerometer_data = create_test_accelerometer_data(days=7)
    
    # Extract feature
    result = features.activity_volume(accelerometer_data)
    
    # Validate results
    assert 'weekly_activity_count' in result
    assert 'quality_metrics' in result
    assert result['quality_metrics']['overall_quality'] > 0.5
    assert result['weekly_activity_count'] > 0
```

## 📚 Theoretical Foundations

Psyconstruct operationalizes established psychological theories into digital behavioral indicators:

- **Behavioral Activation Theory** (Lewinsohn, 1974)
- **Avoidance Learning Theory** (Mowrer, 1947)  
- **Social Engagement Theory** (House, 1981)
- **Circadian Rhythm Theory** (Aschoff, 1965)
- **Measurement Theory** (Lord & Novick, 1968; Bollen & Lennox, 1991)

---

## 🤝 Contributing

We welcome contributions to Psyconstruct! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/dhanumjayareddybhavanam/psyconstruct.git
cd psyconstruct

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run tests to verify setup
python -m pytest psyconstruct/tests/
```

### Code Style

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

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use Psyconstruct in your research, please cite:

```bibtex
@software{bhavanam2026psyconstruct,
  title={Psyconstruct: A reproducible framework for theory-informed digital phenotyping research},
  author={Bhavanam, Dhanumjaya Reddy},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18792047},
  url={https://doi.org/10.5281/zenodo.18792047}
}
```



## 📞 Support

- **Documentation**: [Full documentation](https://github.com/dhanumjayareddybhavanam/psyconstruct/docs)
- **Issues**: [GitHub Issues](https://github.com/dhanumjayareddybhavanam/psyconstruct/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dhanumjayareddybhavanam/psyconstruct/discussions)
- **Email**: dhanumjayareddybhavanam@gmail.com

## 🗺️ Development Roadmap

- [ ] Multi-participant reliability estimation
- [ ] Advanced statistical validation frameworks
- [ ] Cross-dataset harmonization standards
- [ ] Extended construct library

## Version

**v1.0.1** – Initial public research release
