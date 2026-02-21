"""
Unit tests for config module.

Tests configuration parameter validation, default values,
and feature-specific configuration retrieval.
"""

import pytest
from datetime import timedelta
from psyconstruct.config import (
    PsyconstructConfig,
    DataThresholds,
    ClusteringParameters,
    SleepParameters,
    RollingWindowDefaults,
    ResamplingFrequencies,
    ConstructAggregationWeights,
    ValidationParameters,
    get_config
)


class TestDataThresholds:
    """Test DataThresholds configuration."""
    
    def test_default_values(self):
        """Test default threshold values are reasonable."""
        thresholds = DataThresholds()
        
        assert thresholds.min_gps_points_per_day == 10
        assert thresholds.max_gps_gap_hours == 4.0
        assert thresholds.min_accelerometer_coverage == 0.7
        assert thresholds.max_accelerometer_gap_minutes == 60.0
        assert thresholds.min_communication_days == 1
        assert thresholds.min_screen_coverage == 0.5


class TestClusteringParameters:
    """Test ClusteringParameters configuration."""
    
    def test_default_values(self):
        """Test default clustering parameters."""
        clustering = ClusteringParameters()
        
        assert clustering.clustering_radius_meters == 50.0
        assert clustering.min_cluster_size == 5
        assert clustering.home_cluster_radius_meters == 100.0
        assert clustering.nighttime_hours == (22, 6)


class TestSleepParameters:
    """Test SleepParameters configuration."""
    
    def test_default_values(self):
        """Test default sleep parameters."""
        sleep = SleepParameters()
        
        assert sleep.sleep_minimum_duration_hours == 2.0
        assert sleep.sleep_maximum_duration_hours == 12.0
        assert sleep.screen_off_threshold_minutes == 5.0


class TestRollingWindowDefaults:
    """Test RollingWindowDefaults configuration."""
    
    def test_default_values(self):
        """Test default window configurations."""
        windows = RollingWindowDefaults()
        
        assert windows.daily_window == timedelta(days=1)
        assert windows.weekly_window == timedelta(days=7)
        assert windows.biweekly_window == timedelta(days=14)
        assert windows.rolling_3day == timedelta(days=3)
        assert windows.rolling_7day == timedelta(days=7)
        assert windows.rolling_14day == timedelta(days=14)


class TestResamplingFrequencies:
    """Test ResamplingFrequencies configuration."""
    
    def test_default_values(self):
        """Test default resampling frequencies."""
        resampling = ResamplingFrequencies()
        
        assert resampling.gps_resample_freq == "5T"
        assert resampling.accelerometer_resample_freq == "1T"
        assert resampling.screen_resample_freq == "1T"
        assert resampling.communication_resample_freq is None


class TestConstructAggregationWeights:
    """Test ConstructAggregationWeights configuration."""
    
    def test_default_weights_sum_to_one(self):
        """Test that all construct weights sum to 1.0."""
        weights = ConstructAggregationWeights()
        
        # Test BA weights
        ba_sum = sum(weights.ba_weights.values())
        assert abs(ba_sum - 1.0) < 0.001
        
        # Test AV weights
        av_sum = sum(weights.av_weights.values())
        assert abs(av_sum - 1.0) < 0.001
        
        # Test SE weights
        se_sum = sum(weights.se_weights.values())
        assert abs(se_sum - 1.0) < 0.001
        
        # Test RS weights
        rs_sum = sum(weights.rs_weights.values())
        assert abs(rs_sum - 1.0) < 0.001
    
    def test_custom_weights_validation(self):
        """Test custom weights validation in __post_init__."""
        # Valid custom weights
        custom_weights = {
            "activity_volume": 0.5,
            "location_diversity": 0.3,
            "app_usage_breadth": 0.1,
            "activity_timing_variance": 0.1
        }
        weights = ConstructAggregationWeights(ba_weights=custom_weights)
        assert sum(weights.ba_weights.values()) == 1.0


class TestValidationParameters:
    """Test ValidationParameters configuration."""
    
    def test_default_values(self):
        """Test default validation parameters."""
        validation = ValidationParameters()
        
        assert validation.min_entropy_value == 0.0
        assert validation.max_entropy_value == 10.0
        assert validation.min_radius_value == 0.0
        assert validation.max_radius_value == 20000.0
        assert validation.min_rate_value == 0.0
        assert validation.max_rate_value == 1000.0
        assert validation.min_percentage == 0.0
        assert validation.max_percentage == 1.0


class TestPsyconstructConfig:
    """Test main PsyconstructConfig class."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = PsyconstructConfig()
        
        assert config.data_thresholds is not None
        assert config.clustering is not None
        assert config.sleep is not None
        assert config.windows is not None
        assert config.resampling is not None
        assert config.weights is not None
        assert config.validation is not None
    
    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        custom_thresholds = DataThresholds(min_gps_points_per_day=20)
        config = PsyconstructConfig(data_thresholds=custom_thresholds)
        
        assert config.data_thresholds.min_gps_points_per_day == 20
    
    def test_weight_sum_validation(self):
        """Test that configuration validates weight sums."""
        # Invalid weights that don't sum to 1.0
        invalid_weights = {
            "activity_volume": 0.5,
            "location_diversity": 0.3,
            "app_usage_breadth": 0.1,
            "activity_timing_variance": 0.2  # Sum = 1.1
        }
        
        with pytest.raises(ValueError, match="Behavioral Activation weights must sum to 1.0"):
            PsyconstructConfig(
                weights=ConstructAggregationWeights(ba_weights=invalid_weights)
            )
    
    def test_sleep_duration_validation(self):
        """Test sleep duration parameter validation."""
        invalid_sleep = SleepParameters(
            sleep_minimum_duration_hours=8.0,
            sleep_maximum_duration_hours=6.0  # Min > Max
        )
        
        with pytest.raises(ValueError, match="Sleep minimum duration must be less than maximum duration"):
            PsyconstructConfig(sleep=invalid_sleep)
    
    def test_coverage_validation(self):
        """Test coverage parameter validation."""
        # Invalid accelerometer coverage
        invalid_thresholds = DataThresholds(min_accelerometer_coverage=1.5)
        
        with pytest.raises(ValueError, match="Accelerometer coverage must be between 0 and 1"):
            PsyconstructConfig(data_thresholds=invalid_thresholds)
        
        # Invalid screen coverage
        invalid_thresholds = DataThresholds(min_screen_coverage=-0.1)
        
        with pytest.raises(ValueError, match="Screen coverage must be between 0 and 1"):
            PsyconstructConfig(data_thresholds=invalid_thresholds)
    
    def test_get_feature_config(self):
        """Test feature-specific configuration retrieval."""
        config = PsyconstructConfig()
        
        # Test valid feature
        activity_config = config.get_feature_config("activity_volume")
        assert "window" in activity_config
        assert "min_coverage" in activity_config
        assert "resample_freq" in activity_config
        
        # Test invalid feature
        with pytest.raises(ValueError, match="Unknown feature"):
            config.get_feature_config("invalid_feature")
    
    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = PsyconstructConfig()
        config_dict = config.to_dict()
        
        assert "data_thresholds" in config_dict
        assert "clustering" in config_dict
        assert "sleep" in config_dict
        assert "windows" in config_dict
        assert "resampling" in config_dict
        assert "weights" in config_dict
        assert "validation" in config_dict
        
        # Check that window durations are converted to seconds
        assert isinstance(config_dict["windows"]["daily_window"], float)
        assert config_dict["windows"]["daily_window"] == 86400.0  # 1 day in seconds


class TestGetConfig:
    """Test get_config function."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_config()
        assert isinstance(config, PsyconstructConfig)
        assert config.data_thresholds.min_gps_points_per_day == 10
    
    def test_get_config_with_path(self):
        """Test getting configuration with file path (placeholder)."""
        # Currently returns default config
        config = get_config("nonexistent_path.yaml")
        assert isinstance(config, PsyconstructConfig)


class TestFeatureConfigurations:
    """Test specific feature configurations."""
    
    def test_activity_volume_config(self):
        """Test activity volume feature configuration."""
        config = PsyconstructConfig()
        feature_config = config.get_feature_config("activity_volume")
        
        assert feature_config["window"] == timedelta(days=1)
        assert feature_config["min_coverage"] == 0.7
        assert feature_config["resample_freq"] == "1T"
    
    def test_location_diversity_config(self):
        """Test location diversity feature configuration."""
        config = PsyconstructConfig()
        feature_config = config.get_feature_config("location_diversity")
        
        assert feature_config["window"] == timedelta(days=7)
        assert feature_config["min_points"] == 10
        assert feature_config["clustering_radius"] == 50.0
    
    def test_home_confinement_config(self):
        """Test home confinement feature configuration."""
        config = PsyconstructConfig()
        feature_config = config.get_feature_config("home_confinement")
        
        assert feature_config["home_radius"] == 100.0
        assert feature_config["nighttime_hours"] == (22, 6)
        assert feature_config["min_points"] == 10
    
    def test_sleep_features_config(self):
        """Test sleep-related feature configurations."""
        config = PsyconstructConfig()
        
        for feature in ["sleep_onset_consistency", "sleep_duration", "circadian_midpoint"]:
            feature_config = config.get_feature_config(feature)
            
            assert feature_config["min_duration"] == 2.0
            assert feature_config["max_duration"] == 12.0
            assert feature_config["screen_off_threshold"] == 5.0
