Below is a **complete Product Requirements Document (PRD)** in Markdown format for your Python package implementing the 14 construct-aligned digital phenotyping features.

You can copy this directly into a file named:

```
CONSTRUCT_ALIGNED_DIGITAL_PHENOTYPING_PRD.md
```

---

# Construct-Aligned Digital Phenotyping Toolkit

## Product Requirements Document (PRD)

---

# 1. Overview

## 1.1 Product Name (Working)

`psyconstruct` (placeholder)

## 1.2 Purpose

This package implements 14 theory-grounded smartphone-derived behavioral features aligned with four psychological constructs:

* Behavioral Activation (BA)
* Avoidance (AV)
* Social Engagement (SE)
* Routine Stability (RS)

The toolkit is designed for:

* Research reproducibility
* Transparent feature extraction
* Cross-dataset standardization
* Construct-level interpretability

This package is **not** a prediction engine.
It is a **measurement and feature engineering toolkit**.

## 1.3 Ontological Positioning

This toolkit defines a construct-level ontology for behavioral digital phenotyping.

The package operationalizes psychological constructs through observable digital behavioral indicators, establishing a formal mapping between theoretical constructs and measurable digital phenotypes.

---

# 2. Scientific Design Principles

## 2.1 Measurement Model Assumptions

### Construct Operationalization

* **Behavioral Activation (BA)**: Reflective model - features are assumed to be manifestations of the underlying latent construct
* **Avoidance (AV)**: Reflective model - features reflect the tendency to avoid social/environmental engagement
* **Social Engagement (SE)**: Formative model - construct is formed by the composite of social behaviors
* **Routine Stability (RS)**: Reflective model - features reflect the stability of behavioral patterns

### Feature Loading Assumptions

* Features are assumed to load onto their designated latent variables
* Loading weights are initially set to equal weighting
* Weights can be empirically learned through factor analysis in future validation

### Aggregation Structure

* Default aggregation: Linear (additive) combination of z-scored features
* Equal weighting assumed unless empirically validated otherwise
* Alternative weighting schemes supported through configuration

### Mathematical Representation

Current implementation assumes:
```
BA_score = mean([z(ActivityVolume), z(LocationEntropy), z(AppBreadth), z(ActivityVariance)])
```

This represents:
* Equal weighting (λ = 0.25 for each feature)
* Reflective measurement model
* Additive structure
* Within-person standardization

## 2.2 Core Design Principles

1. Each feature must have:

   * Explicit mathematical definition
   * Clear input schema
   * Assumptions documented
   * Known limitations documented

2. No hidden transformations.

3. No undocumented normalization.

4. No black-box ML in feature computation.

5. Within-participant and between-participant normalization must be optional.

6. All functions must be deterministic.

---

# 3. System Architecture

```
psyconstruct/
│
├── preprocessing/
│   ├── harmonization.py
│   └── temporal_features.py
├── features/
├── constructs/
│   ├── aggregator.py
│   └── registry.yaml
├── validation/
├── analysis/
│   ├── reliability.py
│   ├── factor_analysis.py
│   └── mixed_models.py
├── utils/
├── tests/
└── examples/
```

---

# 4. Standard Input Data Schema

All modules must enforce strict schema validation.

## 4.1 GPS Data

| Column    | Type     | Description     |
| --------- | -------- | --------------- |
| timestamp | datetime | UTC ISO format  |
| latitude  | float    | Decimal degrees |
| longitude | float    | Decimal degrees |

---

## 4.2 Accelerometer Data

| Column    | Type     | Description |
| --------- | -------- | ----------- |
| timestamp | datetime | UTC         |
| x         | float    | Accel axis  |
| y         | float    | Accel axis  |
| z         | float    | Accel axis  |

Magnitude must be computed as:

[
\sqrt{x^2 + y^2 + z^2}
]

---

## 4.3 Communication Logs

| Column     | Type     | Description       |
| ---------- | -------- | ----------------- |
| timestamp  | datetime | UTC               |
| direction  | string   | "in" or "out"     |
| contact_id | string   | Unique identifier |

---

## 4.4 Screen State Data

| Column    | Type     | Description   |
| --------- | -------- | ------------- |
| timestamp | datetime | UTC           |
| state     | string   | "on" or "off" |

---

# 5. Module Specifications (14 Features)

---

# MODULE 1: Behavioral Activation (BA)

File: `features/behavioral_activation.py`

---

## BA-1 Activity Volume

### Definition

Rolling sum of accelerometer magnitude over 24-hour window.

[
AV = \sum_{t \in 24h} magnitude_t
]

### Inputs

* accelerometer dataframe
* window length (default = 24H)

### Output

* pandas Series indexed by date

### Edge Cases

* Missing sampling intervals
* Device frequency variability

---

## BA-2 Location Diversity

### Definition

Shannon entropy of clustered GPS locations per week.

[
H = -\sum p_i \log p_i
]

### Requirements

* Pre-clustered location labels
* Minimum data threshold per day

### Output

Weekly entropy score.

---

## BA-3 App Usage Breadth

### Definition

Entropy of app category usage per day.

Requires:

* App category mapping
* Count per category

### Output

Daily entropy value.

---

## BA-4 Activity Timing Variance

### Definition

Variance of peak activity hour across 7–14 days.

Steps:

1. Compute hourly movement totals.
2. Identify peak hour per day.
3. Compute variance.

---

# MODULE 2: Avoidance (AV)

File: `features/avoidance.py`

---

## AV-1 Home Confinement

### Definition

Percentage of GPS readings within defined home cluster radius.

Steps:

1. Identify nighttime dominant cluster.
2. Compute proportion within cluster.

---

## AV-2 Communication Gaps

### Definition

Maximum duration without outgoing communication per day.

Output:

* Daily longest silence (hours)

---

## AV-3 Movement Radius

### Definition

Radius of gyration:

[
r_g = \sqrt{\frac{1}{N} \sum (r_i - r_{cm})^2}
]

Where:

* ( r_{cm} ) = center of mass of coordinates
* Distances computed using haversine

Output:

* Weekly radius value

---

# MODULE 3: Social Engagement (SE)

File: `features/social_engagement.py`

---

## SE-1 Communication Frequency

Count of outgoing communications per day.

---

## SE-2 Contact Diversity

Unique contacts per rolling 7-day window.

---

## SE-3 Initiation Rate

[
IR = \frac{Outgoing}{Outgoing + Incoming}
]

Must handle:

* Zero division
* Sparse communication days

---

# MODULE 4: Routine Stability (RS)

File: `features/routine_stability.py`

---

## RS-1 Sleep Onset Consistency

### Method

1. Detect longest continuous screen-off interval per night.
2. Mark as sleep onset.
3. Compute SD across 14 days.

---

## RS-2 Sleep Duration

Average length of inferred sleep interval.

---

## RS-3 Activity Fragmentation

Entropy of hourly activity distribution within a day.

---

## RS-4 Circadian Midpoint

Midpoint between inferred sleep onset and wake time.

---

# 6. Construct Aggregator Module

File: `constructs/aggregator.py`

---

## Responsibilities

* Z-score features (optional)
* Within-participant normalization
* Weighted or unweighted averaging
* Output construct-level score
* Metadata and provenance tracking

Example:

```
BA_score = mean([z(ActivityVolume), z(LocationEntropy), ...])
```

## Provenance Tracking

* Feature extraction log with timestamps
* Parameter snapshot for each computation
* Version tag embedded in output
* Random seed registry (deterministic operations)

---

# 7. Data Harmonization Layer

File: `preprocessing/harmonization.py`

---

## Purpose

Standardize cross-platform data inconsistencies before feature extraction.

## Core Functions

### Resampling Strategy

* GPS: Resample to 5-minute intervals using median aggregation
* Accelerometer: Resample to 1-minute intervals using mean magnitude
* Communication: No resampling (event-based)
* Screen State: Resample to 1-minute intervals using mode

### Timezone Harmonization

* Convert all timestamps to UTC
* Store original timezone as metadata
* Day boundary adjustment for local time analysis

### Temporal Segmentation

* Weekend/weekday flag generation
* Work hours vs. non-work hours segmentation
* Holiday detection (optional calendar integration)

### Missing Data Policy

* Minimum data threshold: 70% coverage required per day
* Maximum gap tolerance: 4 hours for continuous features
* Imputation strategy: Linear interpolation for gaps < 1 hour
* Missingness flagging for all imputed values

### Device Metadata Handling

* Device type classification (iOS/Android)
* Sampling frequency normalization
* Sensor-specific bias correction

---

# 8. Temporal Modeling Support

File: `preprocessing/temporal_features.py`

---

## Longitudinal Feature Extraction

### Sliding Window Features

* 3-day, 7-day, 14-day rolling windows
* Overlapping windows with configurable step size
* Window-specific feature statistics

### Lagged Features

* Day-to-day change scores
* Autocorrelation metrics
* Cross-lagged correlations between constructs

### Change Point Detection

* Breakpoint detection hooks for behavioral changes
* Statistical process control charts
* Trend analysis utilities

### Rolling Normalization

* Rolling z-scores with configurable windows
* Person-specific baseline establishment
* Deviation from personal norms

---

# 9. Construct Registry

File: `constructs/registry.yaml`

---

## Structure

```yaml
constructs:
  behavioral_activation:
    features:
      - name: activity_volume
        weight: 0.25
        aggregation: mean
        validation_status: theoretical
      - name: location_diversity
        weight: 0.25
        aggregation: mean
        validation_status: theoretical
      - name: app_usage_breadth
        weight: 0.25
        aggregation: mean
        validation_status: experimental
      - name: activity_timing_variance
        weight: 0.25
        aggregation: mean
        validation_status: theoretical
    measurement_model: reflective
    aggregation_type: linear
  avoidance:
    features:
      - name: home_confinement
        weight: 0.33
        aggregation: mean
        validation_status: theoretical
      - name: communication_gaps
        weight: 0.33
        aggregation: mean
        validation_status: theoretical
      - name: movement_radius
        weight: 0.34
        aggregation: inverse_mean
        validation_status: theoretical
    measurement_model: reflective
    aggregation_type: linear
  social_engagement:
    features:
      - name: communication_frequency
        weight: 0.33
        aggregation: mean
        validation_status: validated
      - name: contact_diversity
        weight: 0.33
        aggregation: mean
        validation_status: validated
      - name: initiation_rate
        weight: 0.34
        aggregation: mean
        validation_status: theoretical
    measurement_model: formative
    aggregation_type: linear
  routine_stability:
    features:
      - name: sleep_onset_consistency
        weight: 0.25
        aggregation: inverse_sd
        validation_status: theoretical
      - name: sleep_duration
        weight: 0.25
        aggregation: mean
        validation_status: theoretical
      - name: activity_fragmentation
        weight: 0.25
        aggregation: inverse_entropy
        validation_status: experimental
      - name: circadian_midpoint
        weight: 0.25
        aggregation: circular_sd
        validation_status: theoretical
    measurement_model: reflective
    aggregation_type: linear
```

---

# 10. Statistical Analysis Module

Directory: `analysis/`

---

## File: `analysis/reliability.py`

### Functions

* `cronbach_alpha()` - Internal consistency for construct scales
* `omega_h()` - Hierarchical omega for multidimensional constructs
* `test_retest_reliability()` - ICC for temporal stability
* `split_half_reliability()` - Alternative reliability estimate

## File: `analysis/factor_analysis.py`

### Functions

* `exploratory_factor_analysis()` - EFA for construct validation
* `confirmatory_factor_analysis()` - CFA for measurement model testing
* `factor_score_extraction()` - Multiple methods for factor scores
* `measurement_invariance()` - Cross-group invariance testing

## File: `analysis/mixed_models.py`

### Functions

* `longitudinal_mixed_effects()` - Growth curve modeling
* `random_intercept_slope()` - Individual differences modeling
* `time_varying_covariates()` - Dynamic predictor modeling
* `multilevel_reliability()` - Hierarchical reliability estimation

---

# 11. Construct Validation Plan

---

## 20. Construct Validation Plan

### Convergent Validity Expectations

* **Behavioral Activation**: Moderate positive correlation with established activity measures (r > .40)
* **Avoidance**: Negative correlation with approach-oriented behaviors (r < -.30)
* **Social Engagement**: Positive correlation with self-reported social support (r > .35)
* **Routine Stability**: Positive correlation with circadian rhythm questionnaires (r > .30)

### Discriminant Validity Expectations

* Inter-construct correlations should be moderate (|r| < .60)
* Each construct should show distinct patterns across clinical subgroups
* Factor analysis should reveal four-factor structure

### Factor Analysis Plan

#### Exploratory Phase

* Principal axis factoring with oblique rotation
* Parallel analysis for factor retention
* Factor loading threshold: |λ| > .40
* Cross-validation with split-sample approach

#### Confirmatory Phase

* Structural equation modeling with maximum likelihood
* Fit indices: CFI > .95, RMSEA < .06, SRMR < .08
* Measurement invariance across demographic groups
* Model comparison with alternative factor structures

### Reliability Metrics

#### Internal Consistency

* Cronbach's α > .70 for each construct scale
* McDonald's ω > .70 for hierarchical structures
* Item-total correlations > .30

#### Test-Retest Reliability

* ICC(2,1) > .75 for 2-week stability
* Standard error of measurement < 0.5 SD
* Minimal detectable change calculation

#### Sensitivity to Change

* Standardized response mean > .50 for clinical change detection
* Reliable change index thresholds
* Effect size benchmarks for meaningful change

### Validation Dataset Requirements

* Minimum N = 300 for factor analysis
* 14-day continuous monitoring
* Multi-modal sensor data collection
* Clinical assessment battery for criterion validation

### Statistical Power Analysis

* Factor analysis: 10 participants per estimated parameter
* Reliability: N > 100 for stable estimates
* Validity: N > 200 for correlation stability

---

# 12. Validation Module

File: `validation/sanity_checks.py`

---

Must include:

* Missing data detection
* Timezone consistency checks
* Sampling frequency validation
* Minimum required days check
* Entropy sanity bounds
* Radius non-negativity checks
* Construct score range validation
* Feature-construct alignment verification

---

# 13. Synthetic Data Generator

File: `validation/synthetic_profiles.py`

Functions:

* generate_control_profile()
* generate_low_activation_profile()
* generate_high_avoidance_profile()

Purpose:

* Internal testing
* Demonstration
* Reproducibility validation

---

# 14. Configuration Layer

File: `config.py`

Must define:

* Minimum GPS points per day
* Clustering radius threshold
* Sleep minimum duration
* Rolling window defaults
* Missing data thresholds
* Resampling frequencies
* Construct aggregation weights

All parameters must be user-overridable.

---

# 15. Technical Stack

## Core Dependencies

* pandas
* numpy
* scipy
* scikit-learn (for clustering)
* haversine
* pytz
* pingouin (for statistical analysis)
* semopy (for SEM/CFA)
* pyyaml (for registry management)

## Testing

* pytest
* coverage

## Documentation

* Sphinx or MkDocs

## Packaging

* pyproject.toml
* setuptools or hatch

## CI/CD

* GitHub Actions
* Automated testing on push

---

# 16. Performance Requirements

* Must handle 6 months of data per user
* Must process daily-level features under 5 seconds per participant
* No unnecessary loops (vectorized operations required)
* Memory-efficient processing for longitudinal analysis

---

# 17. Security & Privacy Considerations

* No data transmission
* Local processing only
* No logging of raw GPS coordinates unless explicitly enabled
* User warning for sensitive data usage
* Provenance metadata without identifiable information
* Secure handling of device-specific identifiers

---

# 18. Testing Requirements

Each feature must include:

* Unit test with synthetic input
* Edge case test
* Boundary test
* Missing data test
* Construct validation test
* Temporal stability test

---

# 19. Documentation Requirements

README must include:

* Theoretical foundation
* Mathematical definitions
* Input schema
* Example workflow
* Ethical considerations
* Citation instructions
* Measurement model documentation
* Construct validation guidelines

---

# 20. Non-Goals

* No prediction models
* No deep learning
* No clinical claims
* No automatic diagnosis
* No causal inference without additional validation

---

# 21. Versioning Plan

* v0.1 — Core 14 features
* v0.2 — Construct aggregation
* v0.3 — Data harmonization layer
* v0.4 — Temporal modeling support
* v0.5 — Statistical analysis module
* v0.6 — Construct validation suite
* v1.0 — Stable release with validation documentation

---

# 22. Deliverables Checklist

* [ ] All 14 features implemented
* [ ] Data harmonization module
* [ ] Temporal feature extraction
* [ ] Construct registry system
* [ ] Statistical analysis module
* [ ] Unit tests > 80% coverage
* [ ] Construct validation tests
* [ ] Synthetic dataset generator
* [ ] Example notebook
* [ ] API documentation
* [ ] Measurement model documentation
* [ ] PyPI-ready packaging
* [ ] License (MIT or BSD recommended)

---

# 23. Success Criteria

The toolkit is considered complete when:

1. Another researcher can reproduce feature extraction from raw sensor logs.
2. All feature definitions are mathematically documented.
3. Construct scores can be computed reproducibly.
4. Synthetic profiles produce expected construct shifts.
5. Code passes all unit tests.
6. Measurement model assumptions are explicitly documented.
7. Construct validation procedures are established.
8. Cross-platform harmonization is demonstrated.

---

# 24. Future Extensions (Optional)

* Wearable integration
* EMA integration module
* Cross-device calibration layer
* Feature registry plug-in system
* Machine learning construct validation
* Real-time monitoring dashboard
* Clinical decision support interfaces
* Multi-site harmonization protocols

---

# END OF ENHANCED PRD

---

This PRD has been enhanced to meet psychoinformatics publication standards:

✅ **Measurement Model Section** - Formal construct operationalization
✅ **Construct Validation Plan** - Comprehensive validation framework
✅ **Data Harmonization Layer** - Cross-platform standardization
✅ **Temporal Modeling Support** - Longitudinal analysis capabilities
✅ **Metadata + Provenance Tracking** - Scientific reproducibility
✅ **Construct Registry System** - Extensible construct management
✅ **Statistical Analysis Module** - Research-oriented utilities
✅ **Ontological Positioning** - Conceptual framework clarification

If you would like next:

* I can generate a `pyproject.toml` template with enhanced dependencies
* Provide first module code skeleton with harmonization
* Design synthetic depression simulation logic with validation
* Or draft the BRM software paper structure aligned with this enhanced PRD

You now have a **measurement science specification** suitable for academic publication and rigorous psychoinformatics research.
