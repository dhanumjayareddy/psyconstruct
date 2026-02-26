---
title: 'Psyconstruct: A reproducible framework for theory-informed digital phenotyping research'
tags:
  - Python
  - digital phenotyping
  - computational social science
  - psychological measurement
  - behavioral research
authors:
  - name: Dhanumjaya Reddy Bhavanam
    orcid: 0009-0008-0678-8261
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 26 February 2026
bibliography: paper.bib
---

# Summary

Digital phenotyping research faces a critical reproducibility gap: while frameworks like Beiwe, RADAR-base, and mindLAMP provide data collection infrastructure, they lack standardized, theoretically-grounded methods for transforming raw sensor data into psychological construct scores. Researchers must implement custom feature extraction and aggregation pipelines, leading to inconsistent methodologies and irreproducible results across studies.

Psyconstruct addresses this gap by providing a reproducible, open-source framework that explicitly operationalizes selected psychological theories into standardized digital behavioral indicators. The framework bridges raw smartphone sensor data and theoretically meaningful construct scores through a registry-based architecture that ensures consistent digital phenotyping research across laboratories.

Psyconstruct is implemented in Python and designed for integration into research data pipelines.

Source code is available at: https://github.com/dhanumjayareddybhavanam/psyconstruct

DOI: 10.5281/zenodo.18792047

# Statement of Need

The rapid growth of digital phenotyping has created a methodological crisis: researchers collect massive amounts of sensor data but lack standardized methods to transform these data into theoretically meaningful psychological constructs [@torous2019]. Existing platforms focus primarily on data collection rather than measurement, forcing research teams to develop custom pipelines that are rarely shared or validated [@marquez2019].

This methodological gap leads to several critical problems:

1. **Irreproducibility**: Different research teams use different feature extraction and aggregation methods for the same constructs
2. **Theoretical disconnect**: Digital features are often selected without clear theoretical grounding
3. **Black-box methods**: Proprietary algorithms prevent scientific scrutiny and replication
4. **Measurement inconsistency**: Lack of standardized construct definitions across studies

Psyconstruct provides a registry-based computational framework that operationalizes selected psychological constructs through explicit feature-to-construct mappings, deterministic aggregation methods, and comprehensive provenance tracking.

# Registry-Based Architecture

Psyconstruct uses a structured registry that maps psychological theories to specific digital behavioral indicators:

- **Behavioral Activation** (Lewinsohn, 1974): Activity volume, location diversity, app usage breadth, timing variance
- **Avoidance** (Mowrer, 1947): Home confinement, communication gaps, movement radius  
- **Social Engagement** (House, 1981): Communication frequency, contact diversity, initiation rate
- **Routine Stability** (Aschoff, 1965): Sleep onset consistency, sleep duration, activity fragmentation, circadian midpoint

Each construct includes explicit feature definitions, normalization strategies, directional transformations, and versioned metadata.

## Deterministic Aggregation Methods

The framework provides mathematically transparent aggregation methods:

- **Normalization**: Z-score, min-max, and robust scaling options
- **Weighting**: Theory-driven feature weights with documented rationale
- **Quality Assessment**: Comprehensive data quality metrics and provenance tracking
- **Dispersion Estimation**: Weighted standard deviation for uncertainty quantification

## Reproducibility Framework

Psyconstruct ensures reproducible research through:

- **Deterministic algorithms**: No stochastic processes in core calculations
- **Version-locked methods**: Algorithm versions stored in provenance metadata
- **Explicit parameters**: All configuration options documented and exposed
- **Comprehensive testing**: Unit and integration tests for edge cases

# Implementation

Psyconstruct is implemented in Python (>=3.8) and structured into two primary layers:

1. Feature extraction modules
2. Construct aggregation modules

Feature modules compute deterministic behavioral indicators from structured sensor inputs (GPS, accelerometer, communication, screen state, app usage). Each feature returns both value and quality metrics.

Construct aggregation is registry-driven. A YAML-based registry defines construct–feature mappings, weights, and directional transformations. Aggregation is performed via linear weighted combinations following normalization (z-score, min-max, or robust scaling).

All operations are deterministic. Provenance metadata record algorithm version, parameters, and quality metrics. The project includes unit and integration tests to ensure deterministic behavior across environments.

# Usage Example

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

# Extract features from sensor data
ba_results = ba_features.activity_volume(accelerometer_data)
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

# Statistical Assumptions and Limitations

## Key Assumptions

- **Linear Additivity**: Features contribute linearly to construct scores
- **Feature Independence**: No adjustment for multicollinearity between features
- **No Latent Structure**: Constructs treated as weighted composites, not latent variables

## Current Limitations

- **Measurement Models**: In v1.0.0, reflective vs formative distinctions documented but not statistically modeled
- **Single-Participant Reliability**: Reliability estimation requires multi-participant data
- **Validation Status**: Features have varying validation levels (theoretical vs experimental)

# Comparison with Existing Software

| Feature | Beiwe | RADAR-base | mindLAMP | Psyconstruct |
|---------|-------|------------|----------|--------------|
| Data Collection | ✓ | ✓ | ✓ | Not primary focus |
| Feature Extraction | Not primary focus | Limited | Not primary focus | ✓ |
| Theory Mapping | Not primary focus | Not primary focus | Not primary focus | ✓ |
| Construct Aggregation | Not primary focus | Not primary focus | Not primary focus | ✓ |
| Registry-Based Design | Not primary focus | Not primary focus | Not primary focus | ✓ |
| Open Source | ✓ | ✓ | ✓ | ✓ |

# Impact

Psyconstruct enables reproducible digital phenotyping research by providing standardized measurement methods, explicit theoretical grounding for digital features, and transparent open-source algorithms with full documentation.

# Acknowledgements

We thank the open-source community for foundational tools and libraries that made this work possible. We also acknowledge the researchers who have contributed to the theoretical foundations of digital phenotyping.

# References

Toruous, J., Kiang, M. V., Lorme, J., & Onnela, J.-P. (2019). The scientific use of smartphones and the need for rigorous standards. *NPJ Digital Medicine*, 2(1), 1–3.

Marquez, D., Barnett, I., Onnela, J.-P., & Torous, J. (2019). Digital phenotyping for mental health: A systematic review. *NPJ Digital Medicine*, 2(1), 1–11.

Lewinsohn, P. M. (1974). A behavioral theory of depression. *Archives of General Psychiatry*, 31(1), 149–156.

Mowrer, O. H. (1947). On the dual nature of learning: A reinterpretation of 'conditioning' and 'problem-solving'. *Harvard Educational Review*, 17(2), 102–118.

House, J. S. (1981). *Work stress and social support*. Addison-Wesley, Reading, MA.

Aschoff, J. (1965). Circadian rhythms in man. *Science*, 148(3676), 1427–1432.
