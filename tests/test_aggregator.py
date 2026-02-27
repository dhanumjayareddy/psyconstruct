"""
Unit tests for Construct Aggregator module.

Tests cover construct aggregation, normalization methods, quality handling,
missing data scenarios, and export functionality.
"""

import pytest
import json
import tempfile
import os
import math
from datetime import datetime
from pathlib import Path

from psyconstruct.constructs import (
    ConstructAggregator,
    AggregationConfig,
    ConstructScore
)


class TestAggregationConfig:
    """Test AggregationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AggregationConfig()
        
        assert config.normalization_method == "zscore"
        assert config.within_participant == True
        assert config.aggregation_method == "weighted_mean"
        assert config.handle_missing == "exclude"
        assert config.min_features_required == 2
        assert config.min_quality_threshold == 0.5
        assert config.include_feature_scores == True
        assert config.include_quality_metrics == True
        assert config.include_normalization_params == True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AggregationConfig(
            normalization_method="minmax",
            within_participant=False,
            aggregation_method="median",
            min_features_required=3,
            min_quality_threshold=0.7
        )
        
        assert config.normalization_method == "minmax"
        assert config.within_participant == False
        assert config.aggregation_method == "median"
        assert config.min_features_required == 3
        assert config.min_quality_threshold == 0.7


class TestConstructAggregator:
    """Test ConstructAggregator class."""
    
    def create_mock_feature_results(self):
        """Create mock feature extraction results."""
        return {
            "activity_volume": {
                "weekly_activity_count": 1000.0,
                "quality_metrics": {"overall_quality": 0.8}
            },
            "location_diversity": {
                "shannon_entropy": 2.5,
                "quality_metrics": {"overall_quality": 0.9}
            },
            "app_usage_breadth": {
                "daily_breadth": 4.0,
                "quality_metrics": {"overall_quality": 0.85}
            },
            "activity_timing_variance": {
                "timing_variance": 0.2,
                "quality_metrics": {"overall_quality": 0.75}
            },
            "home_confinement": {
                "home_confinement_percentage": 60.0,
                "quality_metrics": {"overall_quality": 0.8}
            },
            "communication_gaps": {
                "max_daily_gap_hours": 5.0,
                "quality_metrics": {"overall_quality": 0.7}
            },
            "movement_radius": {
                "radius_of_gyration_meters": 1500.0,
                "quality_metrics": {"overall_quality": 0.9}
            }
        }
    
    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        config = AggregationConfig()
        aggregator = ConstructAggregator(config=config)
        
        assert aggregator.config == config
        assert hasattr(aggregator, 'construct_registry')
        assert hasattr(aggregator.construct_registry, 'constructs')
    
    def test_aggregator_initialization_with_custom_registry(self):
        """Test aggregator initialization with custom registry."""
        config = AggregationConfig()
        from pathlib import Path
        from psyconstruct.constructs.registry import ConstructRegistry
        
        registry_path = Path(__file__).parent.parent / "constructs" / "registry.json"
        custom_registry = ConstructRegistry(registry_path)
        
        aggregator = ConstructAggregator(
            config=config,
            construct_registry=custom_registry
        )
        
        assert aggregator.config == config
        assert hasattr(aggregator, 'construct_registry')
        assert hasattr(aggregator.construct_registry, 'constructs')
        assert 'behavioral_activation' in aggregator.construct_registry.constructs
    
    def test_aggregate_construct_basic(self):
        """Test basic construct aggregation."""
        config = AggregationConfig(
            normalization_method="none",
            aggregation_method="weighted_mean"
        )
        aggregator = ConstructAggregator(config=config)
        
        feature_results = self.create_mock_feature_results()
        
        # Test behavioral activation aggregation
        score = aggregator.aggregate_construct(
            "behavioral_activation",
            feature_results,
            participant_id="test_participant"
        )
        
        # Check result structure
        assert isinstance(score, ConstructScore)
        assert score.construct_name == "behavioral_activation"
        assert score.participant_id == "test_participant"
        assert isinstance(score.score, float)
        assert isinstance(score.normalized_score, float)
        assert isinstance(score.feature_scores, dict)
        assert isinstance(score.quality_metrics, dict)
        assert isinstance(score.aggregation_parameters, dict)
        assert isinstance(score.timestamp, datetime)
        assert score.interpretation is not None
    
    def test_aggregate_construct_with_normalization(self):
        """Test construct aggregation with normalization."""
        config = AggregationConfig(
            normalization_method="zscore",
            aggregation_method="weighted_mean"
        )
        aggregator = ConstructAggregator(config=config)
        
        feature_results = self.create_mock_feature_results()
        
        # Create reference data
        reference_data = {
            "activity_volume": [800, 900, 1000, 1100, 1200],
            "location_diversity": [2.0, 2.5, 3.0, 3.5, 4.0],
            "app_usage_breadth": [3.0, 3.5, 4.0, 4.5, 5.0],
            "activity_timing_variance": [0.1, 0.15, 0.2, 0.25, 0.3]
        }
        
        score = aggregator.aggregate_construct(
            "behavioral_activation",
            feature_results,
            participant_id="test_participant",
            reference_data=reference_data
        )
        
        # Should have normalized scores
        assert len(score.feature_scores) > 0
        for feature_score in score.feature_scores.values():
            assert isinstance(feature_score, float)
    
    def test_aggregate_construct_insufficient_features(self):
        """Test aggregation with insufficient features."""
        config = AggregationConfig(min_features_required=5)
        aggregator = ConstructAggregator(config=config)
        
        feature_results = self.create_mock_feature_results()
        
        # Remove some features
        limited_features = {k: v for k, v in feature_results.items() if k in ["activity_volume", "location_diversity"]}
        
        with pytest.raises(ValueError, match="Insufficient features"):
            aggregator.aggregate_construct("behavioral_activation", limited_features)
    
    def test_aggregate_construct_low_quality_features(self):
        """Test aggregation with low quality features."""
        config = AggregationConfig(min_quality_threshold=0.9)
        aggregator = ConstructAggregator(config=config)
        
        feature_results = self.create_mock_feature_results()
        
        # Make all features low quality
        for feature in feature_results.values():
            feature["quality_metrics"]["overall_quality"] = 0.5
        
        with pytest.raises(ValueError, match="Insufficient high-quality features"):
            aggregator.aggregate_construct("behavioral_activation", feature_results)
    
    def test_aggregate_construct_invalid_construct(self):
        """Test aggregation with invalid construct name."""
        aggregator = ConstructAggregator()
        feature_results = self.create_mock_feature_results()
        
        with pytest.raises(KeyError, match="Construct not found: invalid_construct"):
            aggregator.aggregate_construct("invalid_construct", feature_results)
    
    def test_aggregate_all_constructs(self):
        """Test batch aggregation of all constructs."""
        config = AggregationConfig(
            normalization_method="none",
            aggregation_method="weighted_mean"
        )
        aggregator = ConstructAggregator(config=config)
        
        feature_results = self.create_mock_feature_results()
        
        construct_scores = aggregator.aggregate_all_constructs(
            feature_results,
            participant_id="test_participant"
        )
        
        # Should return dictionary of construct scores
        assert isinstance(construct_scores, dict)
        assert len(construct_scores) > 0
        
        for construct_name, score in construct_scores.items():
            assert isinstance(score, ConstructScore)
            assert score.construct_name == construct_name
            assert score.participant_id == "test_participant"
    
    def test_normalization_methods(self):
        """Test different normalization methods."""
        feature_results = self.create_mock_feature_results()
        reference_data = {
            "activity_volume": [800, 900, 1000, 1100, 1200],
            "location_diversity": [2.0, 2.5, 3.0, 3.5, 4.0],
            "app_usage_breadth": [3.0, 4.0, 5.0, 6.0, 7.0],
            "activity_timing_variance": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        methods = ["none", "zscore", "minmax", "robust"]
        
        for method in methods:
            config = AggregationConfig(normalization_method=method)
            aggregator = ConstructAggregator(config=config)
            
            score = aggregator.aggregate_construct(
                "behavioral_activation",
                feature_results,
                reference_data=reference_data
            )
            
            assert isinstance(score, ConstructScore)
            assert isinstance(score.normalized_score, float)
    
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        feature_results = self.create_mock_feature_results()
        
        methods = ["weighted_mean", "unweighted_mean", "median"]
        
        for method in methods:
            config = AggregationConfig(aggregation_method=method, normalization_method="none")
            aggregator = ConstructAggregator(config=config)
            
            score = aggregator.aggregate_construct(
                "behavioral_activation",
                feature_results
            )
            
            assert isinstance(score, ConstructScore)
            assert isinstance(score.normalized_score, float)
    
    def test_extract_primary_value(self):
        """Test primary value extraction from feature results."""
        aggregator = ConstructAggregator()
        
        # Test different feature types
        test_cases = [
            ("activity_volume", {"weekly_activity_count": 1000.0}, 1000.0),
            ("location_diversity", {"shannon_entropy": 2.5}, 2.5),
            ("home_confinement", {"home_confinement_percentage": 60.0}, 60.0),
            ("unknown_feature", {"score": 0.5}, 0.5),
            ("unknown_feature", {"value": 0.75}, 0.75)
        ]
        
        for feature_name, feature_result, expected in test_cases:
            result = aggregator._extract_primary_value(feature_result, feature_name)
            assert result == expected
    
    def test_extract_quality_score(self):
        """Test quality score extraction."""
        aggregator = ConstructAggregator()
        
        test_cases = [
            ({"quality_metrics": {"overall_quality": 0.8}}, 0.8),
            ({"quality_score": 0.75}, 0.75),
            ({"quality": 0.9}, 0.9),
            ({"other_field": 0.5}, 1.0)  # Default
        ]
        
        for feature_result, expected in test_cases:
            result = aggregator._extract_quality_score(feature_result)
            assert result == expected
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        aggregator = ConstructAggregator()
        
        # Test with reference data
        reference_data = {"test_feature": [10, 20, 30, 40, 50]}
        result = aggregator._minmax_normalize(30, "test_feature", reference_data)
        
        # Should be (30 - 10) / (50 - 10) = 0.5
        assert result == 0.5
        
        # Test without reference data
        result = aggregator._minmax_normalize(30, "test_feature", None)
        assert result == 30.0  # (30 - 0) / 1.0
    
    def test_robust_normalization(self):
        """Test robust normalization."""
        aggregator = ConstructAggregator()
        
        # Test with reference data
        reference_data = {"test_feature": [10, 20, 30, 40, 50]}
        result = aggregator._robust_normalize(30, "test_feature", reference_data)
        
        # Should be (30 - median) / MAD
        # median = 30, MAD = median(|x-30|) = 10
        expected = (30 - 30) / 10  # = 0
        assert result == expected
        
        # Test without reference data
        result = aggregator._robust_normalize(30, "test_feature", None)
        assert result == 30.0  # (30 - 0) / 1.0
    
    def test_weighted_mean_aggregation(self):
        """Test weighted mean aggregation."""
        aggregator = ConstructAggregator()
        
        feature_values = {"f1": 10.0, "f2": 20.0, "f3": 30.0}
        feature_weights = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
        
        # Test the internal weighted mean calculation via dispersion interval
        result = aggregator._calculate_dispersion_interval(feature_values, feature_weights)
        
        # Calculate expected weighted mean: (10*1 + 20*2 + 30*3) / (1+2+3) = 23.33
        expected_mean = (10*1 + 20*2 + 30*3) / (1 + 2 + 3)
        
        # Calculate expected weighted variance for SD calculation
        weights = [1.0, 2.0, 3.0]
        weighted_variance = sum(w * (v - expected_mean)**2 for v, w in zip([10.0, 20.0, 30.0], weights)) / sum(weights)
        expected_sd = math.sqrt(weighted_variance)
        
        assert result is not None
        assert abs(result[0] - (expected_mean - expected_sd)) < 0.001  # Lower bound
        assert abs(result[1] - (expected_mean + expected_sd)) < 0.001  # Upper bound
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        aggregator = ConstructAggregator()
        
        normalized_features = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
        feature_weights = {"f1": 1.0, "f2": 1.0, "f3": 1.0}
        
        result = aggregator._calculate_dispersion_interval(normalized_features, feature_weights)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] < result[1]
        
        # Calculate expected values manually
        values = [1.0, 2.0, 3.0]
        weights = [1.0, 1.0, 1.0]
        
        # Weighted mean
        weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        
        # Weighted variance
        weighted_variance = sum(w * (v - weighted_mean)**2 for v, w in zip(values, weights)) / sum(weights)
        weighted_sd = math.sqrt(weighted_variance)
        
        # Expected interval should be mean ± weighted SD
        expected_low = weighted_mean - weighted_sd
        expected_high = weighted_mean + weighted_sd
        
        assert abs(result[0] - expected_low) < 0.01
        assert abs(result[1] - expected_high) < 0.01
    
    def test_confidence_interval_insufficient_data(self):
        """Test confidence interval with insufficient data."""
        aggregator = ConstructAggregator()
        
        # Single feature
        normalized_features = {"f1": 1.0}
        feature_weights = {"f1": 1.0}
        
        result = aggregator._calculate_dispersion_interval(normalized_features, feature_weights)
        assert result is None
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        aggregator = ConstructAggregator()
        
        feature_qualities = {"f1": 0.8, "f2": 0.9, "f3": 0.7}
        # Use equal weights since that's what the method expects
        used_features = {"f1": 1.0, "f2": 1.0, "f3": 1.0}  # Use as weights, not values
        
        result = aggregator._calculate_aggregation_quality(feature_qualities, used_features)
        
        assert "overall_quality" in result
        assert "quality_variance" in result
        assert "quality_consistency" in result
        assert "feature_count" in result
        assert "feature_qualities" in result
        
        # Check calculated values - should be simple average since weights are equal
        assert abs(result["overall_quality"] - 0.8) < 0.001  # Allow for floating point precision
        assert abs(result["quality_variance"] - 0.010000000000000007) < 0.001
        assert result["feature_count"] == 3
    
    def test_interpretation_generation(self):
        """Test interpretation generation."""
        aggregator = ConstructAggregator()
        
        # Test behavioral activation interpretations
        high_score_interpretation = aggregator._generate_interpretation(
            "behavioral_activation", 0.8, {"overall_quality": 0.9}
        )
        assert "Elevated behavioral activation" in high_score_interpretation
        
        low_score_interpretation = aggregator._generate_interpretation(
            "behavioral_activation", -0.8, {"overall_quality": 0.9}
        )
        assert "Reduced behavioral activation" in low_score_interpretation
        
        # Test low quality interpretation
        low_quality_interpretation = aggregator._generate_interpretation(
            "behavioral_activation", 0.5, {"overall_quality": 0.3}
        )
        assert "Low quality" in low_quality_interpretation
    
    def test_construct_info_retrieval(self):
        """Test construct information retrieval."""
        aggregator = ConstructAggregator()
        
        info = aggregator.get_construct_info("behavioral_activation")
        
        assert "description" in info
        assert "measurement_model" in info
        assert "features" in info
        assert len(info["features"]) > 0
    
    def test_construct_info_invalid_construct(self):
        """Test construct info retrieval with invalid construct."""
        aggregator = ConstructAggregator()
        
        with pytest.raises(KeyError, match="Construct not found: invalid"):
            aggregator.get_construct_info("invalid")
    
    def test_list_constructs(self):
        """Test listing all constructs."""
        aggregator = ConstructAggregator()
        
        constructs = aggregator.list_constructs()
        
        assert isinstance(constructs, list)
        assert len(constructs) > 0
        assert "behavioral_activation" in constructs
        assert "avoidance" in constructs
        assert "social_engagement" in constructs
        assert "routine_stability" in constructs


class TestExportFunctionality:
    """Test export functionality."""
    
    def create_test_scores(self):
        """Create test construct scores for export testing."""
        scores = {}
        
        # Behavioral Activation score
        ba_score = ConstructScore(
            construct_name="behavioral_activation",
            score=100.0,
            normalized_score=0.5,
            feature_scores={"activity_volume": 0.6, "location_diversity": 0.4},
            quality_metrics={"overall_quality": 0.8},
            aggregation_parameters={"method": "weighted_mean"},
            timestamp=datetime.now(),
            participant_id="test_participant",
            interpretation="Test interpretation"
        )
        scores["behavioral_activation"] = ba_score
        
        # Avoidance score
        avoidance_score = ConstructScore(
            construct_name="avoidance",
            score=50.0,
            normalized_score=-0.2,
            feature_scores={"home_confinement": -0.1, "communication_gaps": -0.3},
            quality_metrics={"overall_quality": 0.7},
            aggregation_parameters={"method": "weighted_mean"},
            timestamp=datetime.now(),
            participant_id="test_participant",
            interpretation="Test interpretation"
        )
        scores["avoidance"] = avoidance_score
        
        return scores
    
    def test_json_export(self):
        """Test JSON export functionality."""
        aggregator = ConstructAggregator()
        scores = self.create_test_scores()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export to JSON
            aggregator.export_scores(scores, temp_path, format="json")
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert isinstance(exported_data, dict)
            assert "behavioral_activation" in exported_data
            assert "avoidance" in exported_data
            
            # Check structure of exported data
            ba_exported = exported_data["behavioral_activation"]
            assert "score" in ba_exported
            assert "normalized_score" in ba_exported
            assert "feature_scores" in ba_exported
            assert "quality_metrics" in ba_exported
            assert ba_exported["score"] == 100.0
            assert ba_exported["normalized_score"] == 0.5
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_csv_export(self):
        """Test CSV export functionality."""
        aggregator = ConstructAggregator()
        scores = self.create_test_scores()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export to CSV
            aggregator.export_scores(scores, temp_path, format="csv")
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Read and verify CSV content
            with open(temp_path, 'r') as f:
                content = f.read()
            
            lines = content.strip().split('\n')
            
            # Should have header + 2 data lines
            assert len(lines) == 3
            
            # Check header
            header = lines[0]
            assert "construct" in header
            assert "score" in header
            assert "normalized_score" in header
            assert "overall_quality" in header
            
            # Check data lines
            assert "behavioral_activation" in lines[1]
            assert "avoidance" in lines[2]
            assert "100.0" in lines[1]
            assert "50.0" in lines[2]
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_invalid_format(self):
        """Test export with invalid format."""
        aggregator = ConstructAggregator()
        scores = self.create_test_scores()
        
        with tempfile.NamedTemporaryFile() as f:
            temp_path = f.name
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            aggregator.export_scores(scores, temp_path, format="invalid")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_feature_results(self):
        """Test aggregation with empty feature results."""
        aggregator = ConstructAggregator()
        
        with pytest.raises(ValueError, match="Insufficient features"):
            aggregator.aggregate_construct("behavioral_activation", {})
    
    def test_all_features_low_quality(self):
        """Test aggregation when all features are low quality."""
        aggregator = ConstructAggregator()
        
        feature_results = {
            "activity_volume": {
                "weekly_activity_count": 1000.0,
                "quality_metrics": {"overall_quality": 0.3}
            },
            "location_diversity": {
                "shannon_entropy": 2.5,
                "quality_metrics": {"overall_quality": 0.2}
            }
        }
        
        with pytest.raises(ValueError, match="Insufficient high-quality features"):
            aggregator.aggregate_construct("behavioral_activation", feature_results)
    
    def test_mixed_quality_features(self):
        """Test aggregation with mixed quality features."""
        config = AggregationConfig(min_quality_threshold=0.6, normalization_method="none")
        aggregator = ConstructAggregator(config=config)
        
        feature_results = {
            "activity_volume": {
                "weekly_activity_count": 1000.0,
                "quality_metrics": {"overall_quality": 0.8}  # High quality
            },
            "location_diversity": {
                "shannon_entropy": 2.5,
                "quality_metrics": {"overall_quality": 0.4}  # Low quality
            },
            "app_usage_breadth": {
                "daily_breadth": 4.0,
                "quality_metrics": {"overall_quality": 0.9}  # High quality
            }
        }
        
        # Should succeed with high-quality features only
        score = aggregator.aggregate_construct("behavioral_activation", feature_results)
        
        assert isinstance(score, ConstructScore)
        # Should only include high-quality features
        assert "activity_volume" in score.feature_scores
        assert "app_usage_breadth" in score.feature_scores
        assert "location_diversity" not in score.feature_scores
    
    def test_zero_division_in_normalization(self):
        """Test normalization handling zero division."""
        aggregator = ConstructAggregator()
        
        # Test z-score with zero standard deviation
        reference_data = {"test_feature": [10, 10, 10]}  # All same value
        result = aggregator._zscore_normalize(10, "test_feature", reference_data)
        
        # Should handle zero division gracefully
        assert result == 0.0
        
        # Test min-max with zero range
        reference_data = {"test_feature": [10, 10, 10]}
        result = aggregator._minmax_normalize(10, "test_feature", reference_data)
        
        # Should handle zero range gracefully
        assert result == 0.0
