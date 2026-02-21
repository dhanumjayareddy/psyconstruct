"""
Configuration module for psyconstruct package.

This module defines all configurable parameters for digital phenotyping
feature extraction, data harmonization, and construct aggregation.

Product: Construct-Aligned Digital Phenotyping Toolkit
Purpose: Research reproducibility and transparent feature extraction
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import timedelta


@dataclass
class DataThresholds:
    """Minimum data requirements for valid feature extraction."""
    
    # GPS requirements
    min_gps_points_per_day: int = 10
    max_gps_gap_hours: float = 4.0
    
    # Accelerometer requirements  
    min_accelerometer_coverage: float = 0.7  # 70% coverage required
    max_accelerometer_gap_minutes: float = 60.0
    
    # Communication requirements
    min_communication_days: int = 1
    
    # Screen state requirements
    min_screen_coverage: float = 0.5  # 50% coverage required


@dataclass
class ClusteringParameters:
    """Parameters for location clustering and analysis."""
    
    # Location clustering
    clustering_radius_meters: float = 50.0
    min_cluster_size: int = 5
    
    # Home detection
    home_cluster_radius_meters: float = 100.0
    nighttime_hours: tuple = (22, 6)  # 10 PM to 6 AM


@dataclass
class SleepParameters:
    """Parameters for sleep detection from screen state."""
    
    sleep_minimum_duration_hours: float = 2.0
    sleep_maximum_duration_hours: float = 12.0
    screen_off_threshold_minutes: float = 5.0


@dataclass
class RollingWindowDefaults:
    """Default window configurations for temporal analysis."""
    
    # Daily features
    daily_window: timedelta = timedelta(days=1)
    
    # Weekly features  
    weekly_window: timedelta = timedelta(days=7)
    
    # Biweekly features
    biweekly_window: timedelta = timedelta(days=14)
    
    # Rolling windows for temporal modeling
    rolling_3day: timedelta = timedelta(days=3)
    rolling_7day: timedelta = timedelta(days=7)
    rolling_14day: timedelta = timedelta(days=14)


@dataclass
class ResamplingFrequencies:
    """Resampling frequencies for different sensor types."""
    
    # GPS resampling
    gps_resample_freq: str = "5T"  # 5 minutes
    
    # Accelerometer resampling
    accelerometer_resample_freq: str = "1T"  # 1 minute
    
    # Screen state resampling
    screen_resample_freq: str = "1T"  # 1 minute
    
    # Communication logs (event-based, no resampling)
    communication_resample_freq: Optional[str] = None


@dataclass
class ConstructAggregationWeights:
    """Default weights for construct aggregation."""
    
    # Behavioral Activation (BA) - equal weighting
    ba_weights: Dict[str, float] = None
    
    # Avoidance (AV) - equal weighting  
    av_weights: Dict[str, float] = None
    
    # Social Engagement (SE) - equal weighting
    se_weights: Dict[str, float] = None
    
    # Routine Stability (RS) - equal weighting
    rs_weights: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default equal weights if not provided."""
        if self.ba_weights is None:
            self.ba_weights = {
                "activity_volume": 0.25,
                "location_diversity": 0.25, 
                "app_usage_breadth": 0.25,
                "activity_timing_variance": 0.25
            }
        
        if self.av_weights is None:
            self.av_weights = {
                "home_confinement": 0.33,
                "communication_gaps": 0.33,
                "movement_radius": 0.34
            }
            
        if self.se_weights is None:
            self.se_weights = {
                "communication_frequency": 0.33,
                "contact_diversity": 0.33,
                "initiation_rate": 0.34
            }
            
        if self.rs_weights is None:
            self.rs_weights = {
                "sleep_onset_consistency": 0.25,
                "sleep_duration": 0.25,
                "activity_fragmentation": 0.25,
                "circadian_midpoint": 0.25
            }


@dataclass
class ValidationParameters:
    """Parameters for data validation and sanity checks."""
    
    # Entropy bounds
    min_entropy_value: float = 0.0
    max_entropy_value: float = 10.0
    
    # Radius bounds
    min_radius_value: float = 0.0
    max_radius_value: float = 20000.0  # 20km max radius
    
    # Rate bounds
    min_rate_value: float = 0.0
    max_rate_value: float = 1000.0
    
    # Percentage bounds
    min_percentage: float = 0.0
    max_percentage: float = 1.0


class PsyconstructConfig:
    """
    Main configuration class for psyconstruct package.
    
    This class centralizes all configurable parameters and provides
    validation methods to ensure parameter consistency.
    
    Attributes:
        data_thresholds: Minimum data requirements
        clustering: Location clustering parameters
        sleep: Sleep detection parameters
        windows: Rolling window configurations
        resampling: Resampling frequencies
        weights: Construct aggregation weights
        validation: Validation parameter bounds
    """
    
    def __init__(self, 
                 data_thresholds: Optional[DataThresholds] = None,
                 clustering: Optional[ClusteringParameters] = None,
                 sleep: Optional[SleepParameters] = None,
                 windows: Optional[RollingWindowDefaults] = None,
                 resampling: Optional[ResamplingFrequencies] = None,
                 weights: Optional[ConstructAggregationWeights] = None,
                 validation: Optional[ValidationParameters] = None):
        """
        Initialize configuration with default or custom parameters.
        
        Args:
            data_thresholds: Data quality thresholds
            clustering: Location clustering parameters
            sleep: Sleep detection parameters
            windows: Rolling window configurations
            resampling: Resampling frequencies
            weights: Construct aggregation weights
            validation: Validation parameter bounds
        """
        self.data_thresholds = data_thresholds or DataThresholds()
        self.clustering = clustering or ClusteringParameters()
        self.sleep = sleep or SleepParameters()
        self.windows = windows or RollingWindowDefaults()
        self.resampling = resampling or ResamplingFrequencies()
        self.weights = weights or ConstructAggregationWeights()
        self.validation = validation or ValidationParameters()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters for consistency.
        
        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate weight sums
        for construct_name, weights_dict in [
            ("Behavioral Activation", self.weights.ba_weights),
            ("Avoidance", self.weights.av_weights), 
            ("Social Engagement", self.weights.se_weights),
            ("Routine Stability", self.weights.rs_weights)
        ]:
            weight_sum = sum(weights_dict.values())
            if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point errors
                raise ValueError(
                    f"{construct_name} weights must sum to 1.0, got {weight_sum:.3f}"
                )
        
        # Validate time ranges
        if self.sleep.sleep_minimum_duration_hours >= self.sleep.sleep_maximum_duration_hours:
            raise ValueError(
                "Sleep minimum duration must be less than maximum duration"
            )
        
        if not (0 <= self.data_thresholds.min_accelerometer_coverage <= 1):
            raise ValueError("Accelerometer coverage must be between 0 and 1")
            
        if not (0 <= self.data_thresholds.min_screen_coverage <= 1):
            raise ValueError("Screen coverage must be between 0 and 1")
    
    def get_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """
        Get configuration specific to a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary containing feature-specific configuration
            
        Raises:
            ValueError: If feature name is not recognized
        """
        feature_configs = {
            "activity_volume": {
                "window": self.windows.daily_window,
                "min_coverage": self.data_thresholds.min_accelerometer_coverage,
                "resample_freq": self.resampling.accelerometer_resample_freq
            },
            "location_diversity": {
                "window": self.windows.weekly_window,
                "min_points": self.data_thresholds.min_gps_points_per_day,
                "clustering_radius": self.clustering.clustering_radius_meters
            },
            "app_usage_breadth": {
                "window": self.windows.daily_window,
                "min_entropy": self.validation.min_entropy_value,
                "max_entropy": self.validation.max_entropy_value
            },
            "home_confinement": {
                "home_radius": self.clustering.home_cluster_radius_meters,
                "nighttime_hours": self.clustering.nighttime_hours,
                "min_points": self.data_thresholds.min_gps_points_per_day
            },
            "communication_gaps": {
                "window": self.windows.daily_window,
                "min_days": self.data_thresholds.min_communication_days
            },
            "movement_radius": {
                "window": self.windows.weekly_window,
                "min_points": self.data_thresholds.min_gps_points_per_day,
                "max_gap_hours": self.data_thresholds.max_gps_gap_hours
            },
            "communication_frequency": {
                "window": self.windows.daily_window,
                "min_days": self.data_thresholds.min_communication_days
            },
            "contact_diversity": {
                "window": self.windows.weekly_window,
                "min_days": self.data_thresholds.min_communication_days
            },
            "initiation_rate": {
                "window": self.windows.daily_window,
                "min_days": self.data_thresholds.min_communication_days
            },
            "sleep_onset_consistency": {
                "min_duration": self.sleep.sleep_minimum_duration_hours,
                "max_duration": self.sleep.sleep_maximum_duration_hours,
                "screen_off_threshold": self.sleep.screen_off_threshold_minutes
            },
            "sleep_duration": {
                "min_duration": self.sleep.sleep_minimum_duration_hours,
                "max_duration": self.sleep.sleep_maximum_duration_hours,
                "screen_off_threshold": self.sleep.screen_off_threshold_minutes
            },
            "activity_fragmentation": {
                "window": self.windows.daily_window,
                "min_coverage": self.data_thresholds.min_accelerometer_coverage
            },
            "circadian_midpoint": {
                "min_duration": self.sleep.sleep_minimum_duration_hours,
                "max_duration": self.sleep.sleep_maximum_duration_hours,
                "screen_off_threshold": self.sleep.screen_off_threshold_minutes
            }
        }
        
        if feature_name not in feature_configs:
            raise ValueError(f"Unknown feature: {feature_name}")
            
        return feature_configs[feature_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "data_thresholds": self.data_thresholds.__dict__,
            "clustering": self.clustering.__dict__,
            "sleep": self.sleep.__dict__,
            "windows": {
                "daily_window": self.windows.daily_window.total_seconds(),
                "weekly_window": self.windows.weekly_window.total_seconds(),
                "biweekly_window": self.windows.biweekly_window.total_seconds(),
                "rolling_3day": self.windows.rolling_3day.total_seconds(),
                "rolling_7day": self.windows.rolling_7day.total_seconds(),
                "rolling_14day": self.windows.rolling_14day.total_seconds()
            },
            "resampling": self.resampling.__dict__,
            "weights": {
                "ba_weights": self.weights.ba_weights,
                "av_weights": self.weights.av_weights,
                "se_weights": self.weights.se_weights,
                "rs_weights": self.weights.rs_weights
            },
            "validation": self.validation.__dict__
        }


# Default configuration instance
DEFAULT_CONFIG = PsyconstructConfig()


def get_config(config_path: Optional[str] = None) -> PsyconstructConfig:
    """
    Get configuration instance.
    
    Args:
        config_path: Optional path to custom configuration file
        
    Returns:
        Configuration instance
    """
    if config_path is None:
        return DEFAULT_CONFIG
    
    # TODO: Implement loading from file
    # For now, return default configuration
    return DEFAULT_CONFIG
