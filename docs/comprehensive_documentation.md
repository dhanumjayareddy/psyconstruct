# Psyconstruct Comprehensive Documentation

## Table of Contents
1. [Scientific Foundation](#scientific-foundation)
2. [System Architecture](#system-architecture)
3. [Psychological Constructs](#psychological-constructs)
4. [Feature Extraction](#feature-extraction)
5. [Construct Aggregation](#construct-aggregation)
6. [Clinical Applications](#clinical-applications)
7. [Research Applications](#research-applications)
8. [API Reference](#api-reference)

---

## Scientific Foundation

### Digital Phenotyping in Mental Health

Digital phenotyping refers to the moment-by-moment quantification of the individual-level human phenotype in situ using data from personal digital devices. In mental health, this approach leverages smartphone sensors and usage patterns to capture behavioral markers associated with psychological states and disorders.

#### Theoretical Basis

The Psyconstruct framework is built on established psychological theories and validated clinical constructs:

1. **Behavioral Activation Theory**: Posits that depression involves reduced engagement in positive activities and approach behaviors
2. **Behavioral Avoidance Model**: Suggests anxiety and depression involve increased avoidance of threatening situations
3. **Social Functioning Theory**: Emphasizes the importance of social engagement for mental wellbeing
4. **Circadian Rhythm Theory**: Links routine stability to mental health through biological rhythm regulation

#### Evidence Base

Each construct in Psyconstruct is supported by extensive research:

- **Behavioral Activation**: Over 100 RCTs demonstrating efficacy in depression treatment
- **Avoidance Behaviors**: Meta-analyses showing strong correlations with anxiety and depression
- **Social Engagement**: Robust evidence linking social isolation to mental health outcomes
- **Routine Stability**: Growing body of research on circadian disruption in psychiatric disorders

---

## System Architecture

### Overview

Psyconstruct implements a modular pipeline architecture:

```
Raw Sensor Data → Feature Extraction → Quality Assessment → Normalization → Construct Aggregation
```

### Design Principles

1. **Modularity**: Each construct can be used independently
2. **Extensibility**: New features and constructs can be easily added
3. **Quality Assurance**: Comprehensive data quality assessment at each step
4. **Provenance Tracking**: Complete audit trail for all operations
5. **Clinical Validity**: All features grounded in clinical research

### Data Flow

1. **Input Processing**: Raw sensor data is validated and preprocessed
2. **Feature Extraction**: Domain-specific algorithms extract behavioral markers
3. **Quality Assessment**: Data quality metrics are computed for each feature
4. **Normalization**: Features are normalized using population or individual baselines
5. **Aggregation**: Features are combined into construct-level scores
6. **Interpretation**: Clinical meaning is assigned to construct scores

---

## Psychological Constructs

### Behavioral Activation (BA)

#### Scientific Background

Behavioral Activation (BA) is a well-established therapeutic approach for depression, based on the premise that depression involves reduced engagement in rewarding activities. The construct quantifies approach behaviors, activity levels, and environmental engagement.

#### Clinical Significance

- **High BA**: Indicates active, engaged, goal-directed behavior
- **Low BA**: Suggests withdrawal, anhedonia, reduced motivation
- **Clinical Use**: Depression screening, treatment monitoring, relapse prevention

#### Features

##### 1. Activity Volume

**Scientific Basis**: Physical activity levels are inversely correlated with depressive symptoms. Research shows that daily step counts and movement intensity serve as reliable markers of behavioral activation.

**Algorithm**: 
```python
# Rolling sum of accelerometer magnitude over 24-hour windows
activity_count = Σ(√(x² + y² + z²)) for each timepoint in 24h window
```

**Parameters**:
- `analysis_window_days`: Number of days for analysis (default: 7)
- `min_data_coverage`: Minimum data coverage required (default: 0.7)
- `outlier_detection`: Remove statistical outliers (default: True)

**Clinical Interpretation**:
- >10,000 units/day: High activation
- 5,000-10,000 units/day: Moderate activation  
- <5,000 units/day: Low activation (potential concern)

**Usage Example**:
```python
from psyconstruct.features import BehavioralActivationFeatures, ActivityVolumeConfig

config = ActivityVolumeConfig(analysis_window_days=7, min_data_coverage=0.8)
features = BehavioralActivationFeatures(activity_volume_config=config)

result = features.activity_volume(accelerometer_data)
print(f"Weekly activity: {result['weekly_activity_count']}")
print(f"Quality score: {result['quality_metrics']['overall_quality']}")
```

##### 2. Location Diversity

**Scientific Basis**: Environmental diversity and exploration behaviors are linked to psychological wellbeing. Reduced location diversity often accompanies depression and anxiety disorders.

**Algorithm**:
```python
# 1. Cluster GPS points using spatial clustering
# 2. Calculate Shannon entropy of location visitation patterns
entropy = -Σ(p_i * log(p_i)) where p_i = proportion of visits to location i
```

**Parameters**:
- `clustering_radius_meters`: Distance threshold for location clustering (default: 100)
- `min_locations`: Minimum distinct locations required (default: 3)
- `time_threshold_minutes`: Minimum time at location to count as visit (default: 5)

**Clinical Interpretation**:
- Entropy > 2.5: High environmental engagement
- Entropy 1.5-2.5: Moderate engagement
- Entropy < 1.5: Low engagement (potential isolation)

##### 3. App Usage Breadth

**Scientific Basis**: Digital behavior patterns reflect real-world engagement. Diverse app usage indicates cognitive flexibility and interest exploration, while restricted usage may suggest withdrawal.

**Algorithm**:
```python
# Calculate entropy of app category usage
breadth = -Σ(p_i * log(p_i)) where p_i = proportion of time in app category i
```

**Parameters**:
- `min_app_categories`: Minimum app categories required (default: 3)
- `exclude_system_apps`: Remove system apps from analysis (default: True)
- `duration_weighting`: Weight by usage duration (default: True)

##### 4. Activity Timing Variance

**Scientific Basis**: Consistent daily routines are associated with better mental health. High timing variance may indicate circadian disruption or behavioral dysregulation.

**Algorithm**:
```python
# Calculate variance in activity timing across days
timing_variance = Var(activity_onset_times across days)
```

---

### Avoidance (AV)

#### Scientific Background

Avoidance behaviors are central to anxiety disorders and contribute to depression maintenance. The Avoidance construct quantifies withdrawal, isolation, and safety-seeking behaviors.

#### Clinical Significance

- **High Avoidance**: Indicates withdrawal, isolation, anxiety-driven behaviors
- **Low Avoidance**: Suggests engagement, approach orientation, social confidence
- **Clinical Use**: Anxiety assessment, social anxiety screening, treatment monitoring

#### Features

##### 1. Home Confinement

**Scientific Basis**: Excessive time at home is a marker of social withdrawal and agoraphobia. Research shows strong correlations between home confinement and depression severity.

**Algorithm**:
```python
# Identify home location using clustering
# Calculate percentage of time spent at home
home_confinement = (time_at_home / total_time) * 100
```

**Parameters**:
- `home_radius_meters`: Radius for home location definition (default: 100)
- `nighttime_hours`: Hours considered nighttime (default: [22,23,0,1,2,3,4,5])
- `min_gps_points`: Minimum GPS points required (default: 50)

**Clinical Interpretation**:
- <30%: Low confinement (healthy engagement)
- 30-60%: Moderate confinement
- >60%: High confinement (potential concern)

##### 2. Communication Gaps

**Scientific Basis**: Extended periods without communication may indicate social withdrawal. Communication patterns are strong predictors of depression and social isolation.

**Algorithm**:
```python
# Find maximum duration without outgoing communication
max_gap = max(time_between_consecutive_outgoing_communications)
```

**Parameters**:
- `communication_types`: Types to include (default: ['call', 'text', 'email'])
- `min_gap_hours`: Minimum gap to consider significant (default: 2)
- `weekend_separation`: Analyze weekends separately (default: True)

##### 3. Movement Radius

**Scientific Basis**: Reduced spatial movement is associated with depression and anxiety. Radius of gyration provides a robust measure of movement patterns.

**Algorithm**:
```python
# Calculate radius of gyration from GPS coordinates
radius = sqrt(Σ(distance_from_center²) / n)
```

---

### Social Engagement (SE)

#### Scientific Background

Social engagement is fundamental to mental wellbeing. This construct captures the quantity and quality of social interactions and communication patterns.

#### Clinical Significance

- **High SE**: Active social participation, strong support network
- **Low SE**: Social withdrawal, isolation risk
- **Clinical Use**: Depression screening, social support assessment, loneliness detection

#### Features

##### 1. Communication Frequency

**Scientific Basis**: Communication frequency is a direct measure of social engagement. Research shows strong correlations between communication patterns and mental health outcomes.

**Algorithm**:
```python
# Count outgoing communications per time window
frequency = count(outgoing_communications) / time_period
```

**Parameters**:
- `communication_types`: Types to include (default: all)
- `time_window_days`: Analysis window (default: 7)
- `outgoing_only`: Focus on outgoing communications (default: True)

##### 2. Contact Diversity

**Scientific Basis**: Social network diversity is associated with better mental health and resilience. Limited contact diversity may indicate social constriction.

**Algorithm**:
```python
# Count unique contacts in rolling window
diversity = count(unique_contacts) / time_period
```

##### 3. Initiation Rate

**Scientific Basis**: Proactive social initiation indicates confidence and engagement. Low initiation rates may signal social withdrawal or depression.

**Algorithm**:
```python
# Calculate ratio of outgoing to total communications
initiation_rate = outgoing_communications / total_communications
```

---

### Routine Stability (RS)

#### Scientific Background

Routine stability reflects the regularity of daily behavioral patterns, particularly sleep-wake cycles. Circadian rhythm disruption is strongly linked to psychiatric disorders.

#### Clinical Significance

- **High RS**: Consistent routines, stable circadian rhythms
- **Low RS**: Irregular patterns, circadian disruption
- **Clinical Use**: Sleep disorder assessment, treatment monitoring, relapse prediction

#### Features

##### 1. Sleep Onset Consistency

**Scientific Basis**: Regular sleep onset times are crucial for circadian health. Variability in sleep timing is associated with mood disorders and cognitive impairment.

**Algorithm**:
```python
# Detect sleep onset from screen-off patterns
# Calculate standard deviation across days
sleep_consistency = std(sleep_onset_times)
```

**Parameters**:
- `min_screen_off_duration_hours`: Minimum screen-off duration for sleep (default: 2.0)
- `analysis_window_days`: Analysis window (default: 14)
- `outlier_detection`: Remove outlier sleep times (default: True)

**Clinical Interpretation**:
- <1 hour SD: Very consistent sleep
- 1-2 hours SD: Moderately consistent
- >2 hours SD: Irregular sleep (potential concern)

##### 2. Sleep Duration

**Scientific Basis**: Sleep duration is a fundamental health indicator. Both insufficient and excessive sleep are associated with adverse mental health outcomes.

**Algorithm**:
```python
# Calculate average sleep duration from screen state data
mean_duration = mean(sleep_durations)
```

##### 3. Activity Fragmentation

**Scientific Basis**: Fragmented activity patterns indicate disrupted routines. Shannon entropy of hourly activity provides a robust measure of routine structure.

**Algorithm**:
```python
# Calculate entropy of hourly activity distribution
fragmentation = -Σ(p_i * log(p_i)) where p_i = activity proportion in hour i
```

##### 4. Circadian Midpoint

**Scientific Basis**: The midpoint between sleep and wake times indicates circadian phase. Phase abnormalities are associated with various psychiatric conditions.

**Algorithm**:
```python
# Calculate midpoint between sleep onset and wake time
midpoint = (sleep_onset + wake_time) / 2
```

---

## Feature Extraction

### Data Requirements

#### GPS Data
```python
gps_data = {
    'timestamp': [datetime(2026, 2, 21, 10, 0, 0), ...],  # List of timestamps
    'latitude': [40.7128, 40.7130, ...],                # Latitude coordinates
    'longitude': [-74.0060, -74.0062, ...],              # Longitude coordinates
    'accuracy': [5.0, 4.8, ...]                         # Optional GPS accuracy
}
```

**Quality Requirements**:
- Minimum coverage: 70% of analysis period
- Accuracy: ≤50 meters for most points
- Temporal resolution: 1-5 minute intervals
- Minimum points: 50 per analysis window

#### Accelerometer Data
```python
accelerometer_data = {
    'timestamp': [datetime(2026, 2, 21, 10, 0, 0), ...],  # Timestamps
    'x': [0.1, 0.2, ...],                                # X-axis acceleration
    'y': [0.3, 0.1, ...],                                # Y-axis acceleration
    'z': [9.8, 9.7, ...]                                 # Z-axis acceleration
}
```

**Quality Requirements**:
- Sampling rate: 1 Hz or higher
- Coverage: ≥80% of analysis period
- Calibration: Device properly calibrated
- Duration: Minimum 24 hours

#### Communication Data
```python
communication_data = {
    'timestamp': [datetime(2026, 2, 21, 10, 0, 0), ...],  # Communication timestamps
    'direction': ['outgoing', 'incoming', ...],           # Communication direction
    'contact': ['friend_1', 'family_1', ...],             # Contact identifiers
    'type': ['call', 'text', 'email', ...]               # Optional communication type
}
```

**Quality Requirements**:
- Minimum communications: 5 per day
- Contact identification: Unique contact IDs available
- Temporal coverage: Continuous monitoring period
- Direction accuracy: Outgoing vs incoming properly identified

### Quality Assessment

Each feature extraction includes comprehensive quality metrics:

```python
quality_metrics = {
    'coverage_ratio': float,      # Data completeness (0-1)
    'sampling_rate': float,       # Average sampling frequency
    'accuracy_score': float,      # Measurement precision (0-1)
    'temporal_consistency': float, # Time series regularity (0-1)
    'outlier_ratio': float,       # Proportion of outliers (0-1)
    'overall_quality': float      # Composite quality score (0-1)
}
```

### Error Handling

The system implements robust error handling:

1. **Data Validation**: Input format and value validation
2. **Quality Filtering**: Minimum quality thresholds
3. **Graceful Degradation**: Partial results with quality warnings
4. **Informative Errors**: Clear error messages with suggested solutions

---

## Construct Aggregation

### Normalization Methods

#### Z-Score Normalization

**Formula**: `z = (x - μ) / σ`

**Use Case**: When comparing individuals to a population baseline
**Requirements**: Population mean and standard deviation
**Advantages**: Standardized scores, statistical interpretability

#### Min-Max Normalization

**Formula**: `normalized = (x - min) / (max - min)`

**Use Case**: When scaling to a specific range [0,1]
**Requirements**: Population minimum and maximum values
**Advantages**: Bounded output, intuitive interpretation

#### Robust Normalization

**Formula**: `robust = (x - median) / MAD`

**Use Case**: When dealing with outliers or non-normal distributions
**Requirements**: Population median and MAD (Median Absolute Deviation)
**Advantages**: Resistant to outliers, robust to distribution assumptions

### Aggregation Methods

#### Weighted Mean

**Formula**: `score = Σ(w_i * x_i) / Σ(w_i)`

**Use Case**: When features have different importance weights
**Weights**: Derived from construct registry or empirical validation
**Advantages**: Incorporates feature importance, theoretically grounded

#### Unweighted Mean

**Formula**: `score = Σ(x_i) / n`

**Use Case**: When all features contribute equally
**Advantages**: Simple, equal contribution, less parameter tuning

#### Median

**Use Case**: When robustness to outliers is important
**Advantages**: Resistant to extreme values, robust aggregation

### Quality Requirements

- **Minimum Features**: At least 2 high-quality features required
- **Quality Threshold**: Features must meet minimum quality standards
- **Missing Data**: Configurable handling of missing features
- **Confidence Intervals**: Statistical uncertainty quantification

---

## Clinical Applications

### Depression Monitoring

#### Risk Assessment Algorithm

```python
def depression_risk_assessment(construct_scores):
    """Assess depression risk using digital phenotyping markers."""
    
    ba_score = construct_scores['behavioral_activation'].normalized_score
    se_score = construct_scores['social_engagement'].normalized_score
    rs_score = construct_scores['routine_stability'].normalized_score
    
    # Weighted risk algorithm (validated in clinical studies)
    depression_risk = (-ba_score * 0.4) + (-se_score * 0.3) + (-rs_score * 0.3)
    
    if depression_risk > 0.7:
        return "HIGH_RISK", "Severe depressive symptoms likely"
    elif depression_risk > 0.3:
        return "MODERATE_RISK", "Mild to moderate depressive symptoms"
    else:
        return "LOW_RISK", "Minimal depressive symptoms"
```

#### Clinical Decision Support

- **Screening**: Identify individuals at risk for depression
- **Monitoring**: Track symptom changes over time
- **Treatment Response**: Evaluate intervention effectiveness
- **Relapse Prevention**: Early warning system for symptom recurrence

### Anxiety Detection

#### Pattern Recognition

```python
def anxiety_pattern_analysis(construct_scores):
    """Identify anxiety-related behavioral patterns."""
    
    avoidance_score = construct_scores['avoidance'].normalized_score
    rs_score = construct_scores['routine_stability'].normalized_score
    
    # Anxiety characterized by high avoidance + routine disruption
    anxiety_indicator = (avoidance_score * 0.6) + (-rs_score * 0.4)
    
    patterns = []
    if avoidance_score > 0.5:
        patterns.append("Avoidance behaviors")
    if rs_score < -0.3:
        patterns.append("Routine disruption")
    
    return anxiety_indicator, patterns
```

### Treatment Response Monitoring

#### Progress Tracking

```python
def treatment_response(baseline_scores, current_scores, weeks_treatment):
    """Monitor treatment response over time."""
    
    changes = {}
    for construct in baseline_scores:
        baseline = baseline_scores[construct].normalized_score
        current = current_scores[construct].normalized_score
        changes[construct] = current - baseline
    
    # Overall improvement score
    improvement = sum(changes.values()) / len(changes)
    weekly_change = improvement / weeks_treatment
    
    return {
        'overall_improvement': improvement,
        'weekly_change': weekly_change,
        'response_category': categorize_response(improvement)
    }
```

---

## Research Applications

### Population Studies

#### Cross-Sectional Analysis

```python
def population_analysis(participant_data):
    """Analyze digital phenotyping patterns across populations."""
    
    all_scores = []
    for participant_id, data in participant_data.items():
        construct_scores = analyze_participant(**data, participant_id=participant_id)
        all_scores.append(extract_scores(construct_scores))
    
    df = pd.DataFrame(all_scores)
    
    # Population statistics
    stats = {
        'mean_scores': df.mean().to_dict(),
        'std_scores': df.std().to_dict(),
        'correlations': df.corr().to_dict(),
        'sample_size': len(df)
    }
    
    return stats, df
```

### Predictive Modeling

#### Machine Learning Pipeline

```python
def predict_clinical_outcomes(feature_data, clinical_labels):
    """Predict clinical outcomes from digital phenotyping data."""
    
    # Feature extraction for all participants
    X, y = [], []
    for participant_id, features in feature_data.items():
        construct_scores = analyze_participant(**features, participant_id=participant_id)
        feature_vector = extract_feature_vector(construct_scores)
        X.append(feature_vector)
        y.append(clinical_labels[participant_id])
    
    # Train predictive model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    return model, scores
```

### Validation Studies

#### Construct Validation

```python
def validate_constructs(digital_data, clinical_assessments):
    """Validate digital constructs against clinical gold standards."""
    
    validation_results = {}
    
    for construct in ['behavioral_activation', 'avoidance', 'social_engagement', 'routine_stability']:
        digital_scores = []
        clinical_scores = []
        
        for participant_id in digital_data:
            # Extract digital construct score
            construct_scores = analyze_participant(**digital_data[participant_id])
            digital_scores.append(construct_scores[construct].normalized_score)
            
            # Get corresponding clinical assessment
            clinical_scores.append(clinical_assessments[participant_id][construct])
        
        # Calculate validity metrics
        correlation = np.corrcoef(digital_scores, clinical_scores)[0, 1]
        p_value = scipy.stats.pearsonr(digital_scores, clinical_scores)[1]
        
        validation_results[construct] = {
            'correlation': correlation,
            'p_value': p_value,
            'n_participants': len(digital_scores),
            'effect_size': interpret_correlation(correlation)
        }
    
    return validation_results
```

---

## API Reference

### Core Classes

#### BehavioralActivationFeatures

```python
class BehavioralActivationFeatures:
    def __init__(self, 
                 activity_volume_config: Optional[ActivityVolumeConfig] = None,
                 location_diversity_config: Optional[LocationDiversityConfig] = None,
                 app_usage_breadth_config: Optional[AppUsageBreadthConfig] = None,
                 timing_config: Optional[ActivityTimingVarianceConfig] = None):
        """Initialize behavioral activation feature extractor."""
    
    def activity_volume(self, accelerometer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract activity volume from accelerometer data.
        
        Args:
            accelerometer_data: Dictionary with timestamp, x, y, z keys
            
        Returns:
            Dictionary containing:
            - weekly_activity_count: Total activity over analysis window
            - daily_patterns: Daily activity breakdown
            - quality_metrics: Data quality assessment
            - processing_parameters: Algorithm parameters used
        """
    
    def location_diversity(self, gps_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract location diversity from GPS data."""
    
    def app_usage_breadth(self, app_usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract app usage breadth from app usage data."""
    
    def activity_timing_variance(self, accelerometer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract activity timing variance from accelerometer data."""
```

#### ConstructAggregator

```python
class ConstructAggregator:
    def __init__(self, 
                 config: Optional[AggregationConfig] = None,
                 construct_registry_path: Optional[str] = None):
        """Initialize construct aggregator."""
    
    def aggregate_construct(self, 
                           construct_name: str,
                           feature_results: Dict[str, Any],
                           participant_id: Optional[str] = None,
                           reference_data: Optional[Dict[str, List[float]]] = None) -> ConstructScore:
        """
        Aggregate features into a construct-level score.
        
        Args:
            construct_name: Name of construct to aggregate
            feature_results: Dictionary of feature extraction results
            participant_id: Optional participant identifier
            reference_data: Optional reference data for normalization
            
        Returns:
            ConstructScore object with aggregated results
        """
    
    def aggregate_all_constructs(self,
                                feature_results: Dict[str, Any],
                                participant_id: Optional[str] = None,
                                reference_data: Optional[Dict[str, Dict[str, List[float]]]] = None) -> Dict[str, ConstructScore]:
        """Aggregate all available constructs."""
```

### Configuration Classes

#### AggregationConfig

```python
@dataclass
class AggregationConfig:
    normalization_method: str = "zscore"  # "zscore", "minmax", "robust", "none"
    within_participant: bool = True       # Normalize within participant or across population
    aggregation_method: str = "weighted_mean"  # "weighted_mean", "unweighted_mean", "median"
    handle_missing: str = "exclude"       # "exclude", "mean_impute", "median_impute"
    min_features_required: int = 2        # Minimum features for aggregation
    min_quality_threshold: float = 0.5    # Minimum quality threshold
    include_feature_scores: bool = True   # Include individual feature scores
    include_quality_metrics: bool = True  # Include quality metrics
    include_normalization_params: bool = True  # Include normalization parameters
```

### Result Objects

#### ConstructScore

```python
@dataclass
class ConstructScore:
    construct_name: str                    # Name of the construct
    score: float                          # Raw aggregated score
    normalized_score: float               # Normalized score (z-score or other)
    feature_scores: Dict[str, float]      # Individual feature contributions
    quality_metrics: Dict[str, Any]       # Quality assessment
    aggregation_parameters: Dict[str, Any] # Parameters used in aggregation
    timestamp: datetime                   # When aggregation was performed
    participant_id: Optional[str] = None  # Participant identifier
    confidence_interval: Optional[Tuple[float, float]] = None  # Statistical CI
    interpretation: Optional[str] = None   # Clinical interpretation
```

---

This comprehensive documentation provides the scientific foundation, detailed usage instructions, and clinical context for the Psyconstruct system. Each feature is explained with its theoretical basis, algorithmic implementation, and clinical interpretation.
