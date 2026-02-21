# API Reference Documentation

## Table of Contents
1. [Core Functions](#core-functions)
2. [Feature Extraction Classes](#feature-extraction-classes)
3. [Construct Aggregation](#construct-aggregation)
4. [Configuration Classes](#configuration-classes)
5. [Data Structures](#data-structures)
6. [Utility Functions](#utility-functions)
7. [Exception Classes](#exception-classes)

---

## Core Functions

### analyze_participant()

**Signature**:
```python
def analyze_participant(
    participant_id: str,
    accelerometer_data: Optional[Dict[str, Any]] = None,
    gps_data: Optional[Dict[str, Any]] = None,
    communication_data: Optional[Dict[str, Any]] = None,
    app_usage_data: Optional[Dict[str, Any]] = None,
    screen_state_data: Optional[Dict[str, Any]] = None,
    config: Optional[AggregationConfig] = None,
    construct_registry_path: Optional[str] = None,
    reference_data: Optional[Dict[str, Dict[str, List[float]]]] = None
) -> Dict[str, ConstructScore]:
```

**Description**: Complete analysis of participant data across all psychological constructs.

**Parameters**:
- `participant_id` (str): Unique identifier for the participant
- `accelerometer_data` (dict, optional): Accelerometer measurements with keys 'timestamp', 'x', 'y', 'z'
- `gps_data` (dict, optional): GPS coordinates with keys 'timestamp', 'latitude', 'longitude', 'accuracy'
- `communication_data` (dict, optional): Communication logs with keys 'timestamp', 'direction', 'contact', 'type'
- `app_usage_data` (dict, optional): App usage information with keys 'timestamp', 'app_name', 'category', 'duration'
- `screen_state_data` (dict, optional): Screen on/off states with keys 'timestamp', 'state'
- `config` (AggregationConfig, optional): Configuration for construct aggregation
- `construct_registry_path` (str, optional): Path to custom construct registry
- `reference_data` (dict, optional): Reference data for population normalization

**Returns**:
- `Dict[str, ConstructScore]`: Dictionary mapping construct names to ConstructScore objects

**Raises**:
- `ValueError`: If insufficient data provided for any construct
- `DataQualityError`: If data quality below minimum thresholds
- `ConfigurationError`: If invalid configuration provided

**Example**:
```python
results = analyze_participant(
    participant_id='user_001',
    accelerometer_data=accel_data,
    gps_data=gps_data,
    communication_data=comm_data,
    app_usage_data=app_data,
    screen_state_data=screen_data
)

for construct_name, score in results.items():
    print(f"{construct_name}: {score.normalized_score:.2f}")
```

---

## Feature Extraction Classes

### BehavioralActivationFeatures

**Description**: Extracts features related to behavioral activation and approach behaviors.

#### Constructor
```python
def __init__(
    self,
    activity_volume_config: Optional[ActivityVolumeConfig] = None,
    location_diversity_config: Optional[LocationDiversityConfig] = None,
    app_usage_breadth_config: Optional[AppUsageBreadthConfig] = None,
    timing_config: Optional[ActivityTimingVarianceConfig] = None
):
```

#### Methods

##### activity_volume()
```python
def activity_volume(self, accelerometer_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Calculates total activity volume from accelerometer data.

**Parameters**:
- `accelerometer_data`: Dictionary with keys 'timestamp', 'x', 'y', 'z'

**Returns**:
```python
{
    'weekly_activity_count': float,           # Total activity over analysis window
    'daily_activity_breakdown': List[float],  # Daily activity counts
    'peak_activity_times': List[datetime],    # Times of peak activity
    'activity_variance': float,               # Variance in activity levels
    'quality_metrics': {
        'coverage_ratio': float,              # Data completeness (0-1)
        'sampling_rate': float,               # Average sampling frequency
        'overall_quality': float              # Composite quality score (0-1)
    },
    'processing_parameters': {
        'analysis_window_days': int,
        'min_data_coverage': float,
        'outlier_detection': bool
    }
}
```

**Raises**:
- `InsufficientDataError`: If insufficient accelerometer data
- `DataQualityError`: If data quality below threshold

##### location_diversity()
```python
def location_diversity(self, gps_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Calculates location diversity and environmental exploration patterns.

**Parameters**:
- `gps_data`: Dictionary with keys 'timestamp', 'latitude', 'longitude', 'accuracy'

**Returns**:
```python
{
    'location_entropy': float,                # Shannon entropy of location visitation
    'unique_locations': int,                  # Number of distinct locations
    'location_frequencies': Dict[str, int],   # Frequency of visits to each location
    'exploration_radius': float,              # Maximum distance from home location
    'home_location': Tuple[float, float],     # Estimated home coordinates
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

##### app_usage_breadth()
```python
def app_usage_breadth(self, app_usage_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Measures diversity of app usage patterns.

**Parameters**:
- `app_usage_data`: Dictionary with keys 'timestamp', 'app_name', 'category', 'duration'

**Returns**:
```python
{
    'category_entropy': float,                # Entropy of app category usage
    'unique_categories': int,                 # Number of app categories used
    'category_distribution': Dict[str, float], # Time distribution across categories
    'app_diversity_score': float,             # Overall diversity metric
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

##### activity_timing_variance()
```python
def activity_timing_variance(self, accelerometer_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Calculates variance in activity timing across days.

**Parameters**:
- `accelerometer_data`: Dictionary with keys 'timestamp', 'x', 'y', 'z'

**Returns**:
```python
{
    'timing_variance': float,                 # Variance in activity onset times
    'daily_onset_times': List[float],         # Activity onset times for each day
    'consistency_score': float,               # Consistency of timing patterns
    'peak_activity_hour': int,                # Hour of peak activity
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

### AvoidanceFeatures

**Description**: Extracts features related to avoidance and withdrawal behaviors.

#### Constructor
```python
def __init__(
    self,
    home_confinement_config: Optional[HomeConfinementConfig] = None,
    communication_gaps_config: Optional[CommunicationGapsConfig] = None,
    movement_radius_config: Optional[MovementRadiusConfig] = None
):
```

#### Methods

##### home_confinement()
```python
def home_confinement(self, gps_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Measures time spent at home vs. other locations.

**Returns**:
```python
{
    'home_confinement_ratio': float,          # Proportion of time at home
    'daytime_home_time': float,               # Time at home during daytime
    'nighttime_home_time': float,             # Time at home during nighttime
    'home_location_confidence': float,        # Confidence in home location estimate
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

##### communication_gaps()
```python
def communication_gaps(self, communication_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Identifies extended periods without communication.

**Returns**:
```python
{
    'max_communication_gap_hours': float,     # Longest period without communication
    'average_gap_hours': float,               # Average time between communications
    'gap_distribution': Dict[str, int],       # Distribution of gap lengths
    'isolated_periods': List[Tuple[datetime, datetime]], # Periods of isolation
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

##### movement_radius()
```python
def movement_radius(self, gps_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Calculates typical movement radius from home location.

**Returns**:
```python
{
    'radius_of_gyration': float,              # Typical movement radius
    'max_distance_from_home': float,          # Maximum distance from home
    'distance_distribution': Dict[str, int],  # Distribution of distances
    'movement_pattern_score': float,          # Overall movement pattern metric
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

### SocialEngagementFeatures

**Description**: Extracts features related to social interaction and engagement.

#### Constructor
```python
def __init__(
    self,
    communication_frequency_config: Optional[CommunicationFrequencyConfig] = None,
    contact_diversity_config: Optional[ContactDiversityConfig] = None,
    initiation_rate_config: Optional[InitiationRateConfig] = None
):
```

#### Methods

##### communication_frequency()
```python
def communication_frequency(self, communication_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Measures frequency and patterns of communication.

**Returns**:
```python
{
    'daily_communication_count': float,       # Average daily communications
    'communication_frequency': float,         # Communications per hour
    'peak_communication_times': List[int],    # Hours of peak communication
    'communication_trend': float,             # Trend over time
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

##### contact_diversity()
```python
def contact_diversity(self, communication_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Measures diversity of social contacts.

**Returns**:
```python
{
    'unique_contacts_count': int,             # Number of unique contacts
    'contact_entropy': float,                 # Entropy of contact distribution
    'contact_frequency_distribution': Dict[str, int], # Frequency per contact
    'social_network_size': int,               # Estimated social network size
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

##### initiation_rate()
```python
def initiation_rate(self, communication_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Calculates rate of initiating communications.

**Returns**:
```python
{
    'initiation_ratio': float,                # Ratio of outgoing to total communications
    'outgoing_frequency': float,              # Outgoing communications per day
    'incoming_frequency': float,              # Incoming communications per day
    'initiation_trend': float,                # Trend in initiation over time
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

### RoutineStabilityFeatures

**Description**: Extracts features related to routine stability and circadian patterns.

#### Constructor
```python
def __init__(
    self,
    sleep_consistency_config: Optional[SleepConsistencyConfig] = None,
    sleep_duration_config: Optional[SleepDurationConfig] = None,
    activity_fragmentation_config: Optional[ActivityFragmentationConfig] = None,
    circadian_midpoint_config: Optional[CircadianMidpointConfig] = None
):
```

#### Methods

##### sleep_onset_consistency()
```python
def sleep_onset_consistency(self, screen_state_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Measures consistency of sleep onset times.

**Returns**:
```python
{
    'sleep_onset_variance': float,            # Variance in sleep onset times
    'average_sleep_onset': float,             # Average sleep onset time (hours)
    'sleep_onset_times': List[float],         # Sleep onset times for each day
    'consistency_score': float,               # Overall consistency metric
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

##### sleep_duration()
```python
def sleep_duration(self, screen_state_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Calculates sleep duration patterns.

**Returns**:
```python
{
    'average_sleep_duration_hours': float,    # Average sleep duration
    'sleep_duration_variance': float,         # Variance in sleep duration
    'sleep_efficiency': float,                # Sleep efficiency metric
    'sleep_duration_distribution': Dict[str, int], # Distribution of sleep durations
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

##### activity_fragmentation()
```python
def activity_fragmentation(self, accelerometer_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Measures fragmentation of daily activity patterns.

**Returns**:
```python
{
    'activity_entropy': float,                # Entropy of hourly activity distribution
    'fragmentation_index': float,             # Overall fragmentation metric
    'hourly_activity_distribution': List[float], # Activity by hour of day
    'routine_stability_score': float,         # Stability of daily patterns
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

##### circadian_midpoint()
```python
def circadian_midpoint(self, screen_state_data: Dict[str, Any]) -> Dict[str, Any]:
```

**Description**: Calculates circadian midpoint from sleep-wake patterns.

**Returns**:
```python
{
    'average_midpoint_hours': float,          # Average circadian midpoint
    'midpoint_variance': float,               # Variance in circadian midpoint
    'midpoint_trend': float,                  # Trend in circadian timing
    'phase_type': str,                        # Chronotype classification
    'quality_metrics': {...},
    'processing_parameters': {...}
}
```

---

## Construct Aggregation

### ConstructAggregator

**Description**: Aggregates features into construct-level scores with normalization and quality assessment.

#### Constructor
```python
def __init__(
    self,
    config: Optional[AggregationConfig] = None,
    construct_registry_path: Optional[str] = None
):
```

#### Methods

##### aggregate_construct()
```python
def aggregate_construct(
    self,
    construct_name: str,
    feature_results: Dict[str, Any],
    participant_id: Optional[str] = None,
    reference_data: Optional[Dict[str, List[float]]] = None
) -> ConstructScore:
```

**Description**: Aggregates features into a single construct score.

**Parameters**:
- `construct_name`: Name of construct to aggregate
- `feature_results`: Dictionary of feature extraction results
- `participant_id`: Optional participant identifier
- `reference_data`: Optional reference data for normalization

**Returns**:
- `ConstructScore`: Aggregated construct score object

**Raises**:
- `ValueError`: If construct name not found in registry
- `InsufficientFeaturesError`: If insufficient features for aggregation
- `NormalizationError`: If normalization fails

##### aggregate_all_constructs()
```python
def aggregate_all_constructs(
    self,
    feature_results: Dict[str, Any],
    participant_id: Optional[str] = None,
    reference_data: Optional[Dict[str, Dict[str, List[float]]]] = None
) -> Dict[str, ConstructScore]:
```

**Description**: Aggregates all available constructs from feature results.

**Returns**:
- `Dict[str, ConstructScore]`: Dictionary of all construct scores

##### get_construct_info()
```python
def get_construct_info(self, construct_name: str) -> Dict[str, Any]:
```

**Description**: Retrieves information about a specific construct.

**Returns**:
```python
{
    'name': str,                              # Construct name
    'description': str,                       # Construct description
    'features': List[str],                    # Associated features
    'weights': Dict[str, float],              # Feature weights
    'normalization_method': str,              # Default normalization method
    'clinical_interpretation': str            # Clinical interpretation guidelines
}
```

##### list_constructs()
```python
def list_constructs(self) -> List[str]:
```

**Description**: Returns list of available constructs.

**Returns**:
- `List[str]`: List of construct names

---

## Configuration Classes

### AggregationConfig

**Description**: Configuration for construct aggregation process.

```python
@dataclass
class AggregationConfig:
    normalization_method: str = "zscore"      # "zscore", "minmax", "robust", "none"
    within_participant: bool = True           # Normalize within participant or across population
    aggregation_method: str = "weighted_mean" # "weighted_mean", "unweighted_mean", "median"
    handle_missing: str = "exclude"           # "exclude", "mean_impute", "median_impute"
    min_features_required: int = 2            # Minimum features for aggregation
    min_quality_threshold: float = 0.5        # Minimum quality threshold
    include_feature_scores: bool = True       # Include individual feature scores
    include_quality_metrics: bool = True      # Include quality metrics
    include_normalization_params: bool = True # Include normalization parameters
```

### Feature Configuration Classes

#### ActivityVolumeConfig
```python
@dataclass
class ActivityVolumeConfig:
    analysis_window_days: int = 7
    min_data_coverage: float = 0.7
    outlier_detection: bool = True
    outlier_threshold: float = 3.0
```

#### LocationDiversityConfig
```python
@dataclass
class LocationDiversityConfig:
    clustering_radius_meters: float = 100.0
    min_locations: int = 3
    time_threshold_minutes: int = 5
    accuracy_threshold_meters: float = 50.0
```

#### HomeConfinementConfig
```python
@dataclass
class HomeConfinementConfig:
    home_radius_meters: float = 100.0
    nighttime_hours: List[int] = field(default_factory=lambda: [22, 23, 0, 1, 2, 3, 4, 5])
    min_gps_points: int = 50
    confidence_threshold: float = 0.8
```

#### SleepConsistencyConfig
```python
@dataclass
class SleepConsistencyConfig:
    min_screen_off_duration_hours: float = 2.0
    analysis_window_days: int = 14
    outlier_detection: bool = True
    outlier_threshold: float = 2.0
```

---

## Data Structures

### ConstructScore

**Description**: Container for construct-level analysis results.

```python
@dataclass
class ConstructScore:
    construct_name: str                       # Name of the construct
    score: float                             # Raw aggregated score
    normalized_score: float                  # Normalized score (z-score or other)
    feature_scores: Dict[str, float]         # Individual feature contributions
    quality_metrics: Dict[str, Any]          # Quality assessment
    aggregation_parameters: Dict[str, Any]   # Parameters used in aggregation
    timestamp: datetime                      # When aggregation was performed
    participant_id: Optional[str] = None     # Participant identifier
    confidence_interval: Optional[Tuple[float, float]] = None  # Statistical CI
    interpretation: Optional[str] = None      # Clinical interpretation
    normalization_parameters: Optional[Dict[str, Any]] = None  # Normalization params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConstructScore':
        """Create from dictionary."""
```

### QualityMetrics

**Description**: Quality assessment for feature extraction and aggregation.

```python
@dataclass
class QualityMetrics:
    coverage_ratio: float                     # Data completeness (0-1)
    sampling_rate: float                      # Average sampling frequency
    accuracy_score: float                     # Measurement precision (0-1)
    temporal_consistency: float               # Time series regularity (0-1)
    outlier_ratio: float                      # Proportion of outliers (0-1)
    overall_quality: float                    # Composite quality score (0-1)
    
    def meets_minimum_threshold(self, threshold: float = 0.5) -> bool:
        """Check if quality meets minimum threshold."""
        
    def get_quality_level(self) -> str:
        """Get quality level category."""
```

---

## Utility Functions

### Data Validation

#### validate_accelerometer_data()
```python
def validate_accelerometer_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate accelerometer data format and quality."""
    
    validation_result = {
        'is_valid': bool,
        'errors': List[str],
        'warnings': List[str],
        'quality_metrics': QualityMetrics
    }
    
    return validation_result
```

#### validate_gps_data()
```python
def validate_gps_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate GPS data format and quality."""
```

#### validate_communication_data()
```python
def validate_communication_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate communication data format and quality."""
```

### Data Processing

#### preprocess_accelerometer_data()
```python
def preprocess_accelerometer_data(
    data: Dict[str, Any],
    remove_outliers: bool = True,
    interpolate_missing: bool = True,
    low_pass_filter: bool = True
) -> Dict[str, Any]:
    """Preprocess accelerometer data for analysis."""
```

#### detect_sleep_periods()
```python
def detect_sleep_periods(
    screen_state_data: Dict[str, Any],
    min_duration_hours: float = 2.0
) -> List[Tuple[datetime, datetime]]:
    """Detect sleep periods from screen state data."""
```

#### cluster_gps_locations()
```python
def cluster_gps_locations(
    gps_data: Dict[str, Any],
    radius_meters: float = 100.0,
    min_points: int = 5
) -> Dict[str, Any]:
    """Cluster GPS points into locations."""
```

### Clinical Interpretation

#### interpret_construct_score()
```python
def interpret_construct_score(
    construct_name: str,
    normalized_score: float,
    confidence_interval: Optional[Tuple[float, float]] = None
) -> str:
    """Generate clinical interpretation of construct score."""
```

#### assess_clinical_risk()
```python
def assess_clinical_risk(
    construct_scores: Dict[str, ConstructScore],
    risk_model: str = 'default'
) -> Dict[str, Any]:
    """Assess clinical risk from construct scores."""
    
    risk_assessment = {
        'overall_risk': str,                  # "LOW", "MODERATE", "HIGH"
        'risk_probability': float,             # Risk probability (0-1)
        'contributing_factors': List[str],     # Contributing constructs
        'recommendations': List[str],          # Clinical recommendations
        'urgency_level': str                   # "ROUTINE", "PRIORITY", "URGENT"
    }
    
    return risk_assessment
```

### Export and Import

#### export_results()
```python
def export_results(
    construct_scores: Dict[str, ConstructScore],
    format: str = 'json',
    output_path: Optional[str] = None
) -> Union[str, Dict[str, Any]]:
    """Export construct scores to specified format."""
    
    if format == 'json':
        return json.dumps([score.to_dict() for score in construct_scores.values()])
    elif format == 'csv':
        return _export_to_csv(construct_scores, output_path)
    elif format == 'excel':
        return _export_to_excel(construct_scores, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
```

#### import_results()
```python
def import_results(
    file_path: str,
    format: str = 'json'
) -> Dict[str, ConstructScore]:
    """Import construct scores from file."""
```

---

## Exception Classes

### PsyconstructError

**Description**: Base exception class for Psyconstruct.

```python
class PsyconstructError(Exception):
    """Base exception for Psyconstruct package."""
    pass
```

### DataQualityError

```python
class DataQualityError(PsyconstructError):
    """Raised when data quality is insufficient for analysis."""
    
    def __init__(self, message: str, quality_metrics: Optional[QualityMetrics] = None):
        super().__init__(message)
        self.quality_metrics = quality_metrics
```

### InsufficientDataError

```python
class InsufficientDataError(PsyconstructError):
    """Raised when insufficient data is provided for analysis."""
    
    def __init__(self, message: str, required_data: Optional[str] = None):
        super().__init__(message)
        self.required_data = required_data
```

### ConfigurationError

```python
class ConfigurationError(PsyconstructError):
    """Raised when invalid configuration is provided."""
    
    def __init__(self, message: str, config_parameter: Optional[str] = None):
        super().__init__(message)
        self.config_parameter = config_parameter
```

### NormalizationError

```python
class NormalizationError(PsyconstructError):
    """Raised when normalization fails."""
    
    def __init__(self, message: str, normalization_method: Optional[str] = None):
        super().__init__(message)
        self.normalization_method = normalization_method
```

### FeatureExtractionError

```python
class FeatureExtractionError(PsyconstructError):
    """Raised when feature extraction fails."""
    
    def __init__(self, message: str, feature_name: Optional[str] = None):
        super().__init__(message)
        self.feature_name = feature_name
```

---

## Type Hints

### Common Type Definitions

```python
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np

# Data type aliases
AccelerometerData = Dict[str, Any]
GPSData = Dict[str, Any]
CommunicationData = Dict[str, Any]
AppUsageData = Dict[str, Any]
ScreenStateData = Dict[str, Any]

FeatureResults = Dict[str, Any]
ConstructScores = Dict[str, ConstructScore]
ReferenceData = Dict[str, Dict[str, List[float]]]

# Configuration type aliases
NormalizationMethod = str  # "zscore", "minmax", "robust", "none"
AggregationMethod = str    # "weighted_mean", "unweighted_mean", "median"
MissingDataHandling = str  # "exclude", "mean_impute", "median_impute"
```

### Generic Functions

```python
def validate_data_format(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate that data contains required keys."""

def calculate_quality_score(metrics: Dict[str, float]) -> float:
    """Calculate composite quality score from individual metrics."""

def normalize_values(
    values: List[float],
    method: NormalizationMethod,
    reference_params: Optional[Dict[str, float]] = None
) -> List[float]:
    """Normalize values using specified method."""

def aggregate_features(
    features: Dict[str, float],
    weights: Dict[str, float],
    method: AggregationMethod
) -> float:
    """Aggregate features using specified method."""
```

This comprehensive API reference provides detailed documentation for all classes, methods, and functions available in the Psyconstruct package, including parameters, return values, exceptions, and usage examples.
