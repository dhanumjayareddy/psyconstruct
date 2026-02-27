"""
Comprehensive edge case testing for Psyconstruct package.

This module tests edge cases, boundary conditions, and error handling
across all major components of the system.

Product: Construct-Aligned Digital Phenotyping Toolkit
Purpose: Edge case testing and boundary condition validation
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from typing import Dict, Any, List

# Import psyconstruct modules
from psyconstruct.constructs.registry import ConstructRegistry
from psyconstruct.constructs.aggregator import ConstructAggregator, AggregationConfig
from psyconstruct.features.behavioral_activation import BehavioralActivationFeatures
from psyconstruct.features.avoidance import AvoidanceFeatures
from psyconstruct.features.routine_stability import RoutineStabilityFeatures
from psyconstruct.features.social_engagement import SocialEngagementFeatures
from psyconstruct.preprocessing.harmonization import DataHarmonizer, HarmonizationConfig
from psyconstruct.utils.adaptive_quality import AdaptiveQualityAssessor, QualityRegime


class TestRegistryEdgeCases(unittest.TestCase):
    """Test edge cases for the registry system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ConstructRegistry()
    
    def test_empty_feature_values(self):
        """Test handling of empty feature values."""
        aggregator = ConstructAggregator()
        
        with self.assertRaises(ValueError):
            aggregator.aggregate_construct("behavioral_activation", {})
    
    def test_single_feature_construct(self):
        """Test constructs with only one feature."""
        # Create mock single feature data
        feature_results = {
            "activity_volume": {
                "weekly_activity_count": 100,
                "quality": 0.8
            }
        }
        
        aggregator = ConstructAggregator(config=AggregationConfig(normalization_method="none", min_features_required=1), construct_registry=self.registry)
        score = aggregator.aggregate_construct("behavioral_activation", feature_results)
        
        # Should handle single feature gracefully
        self.assertIsNotNone(score)
        self.assertIsInstance(score.score, float)
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields in feature results."""
        feature_results = {
            "activity_volume": {
                # Missing weekly_activity_count
                "quality": 0.8
            }
        }
        
        aggregator = ConstructAggregator(config=AggregationConfig(normalization_method="none", min_features_required=1), construct_registry=self.registry)
        # Should handle missing primary value gracefully by raising error
        with self.assertRaises(ValueError):
            score = aggregator.aggregate_construct("behavioral_activation", feature_results)
    
    def test_invalid_quality_scores(self):
        """Test handling of invalid quality scores."""
        feature_results = {
            "activity_volume": {
                "weekly_activity_count": 100,
                "quality": 1.5  # Invalid quality > 1.0
            },
            "location_diversity": {
                "shannon_entropy": 1.5,
                "quality": -0.1  # Invalid quality < 0.0
            }
        }
        
        aggregator = ConstructAggregator(config=AggregationConfig(normalization_method="none", min_features_required=1), construct_registry=self.registry)
        # Should handle invalid quality scores
        score = aggregator.aggregate_construct("behavioral_activation", feature_results)
        self.assertIsNotNone(score)
    
    def test_extreme_feature_values(self):
        """Test handling of extreme feature values."""
        feature_results = {
            "activity_volume": {
                "weekly_activity_count": 1e10,  # Very large value
                "quality": 0.8
            },
            "location_diversity": {
                "shannon_entropy": 1e-10,  # Very small value
                "quality": 0.8
            }
        }
        
        aggregator = ConstructAggregator(config=AggregationConfig(normalization_method="none", min_features_required=1), construct_registry=self.registry)
        score = aggregator.aggregate_construct("behavioral_activation", feature_results)
        
        # Should handle extreme values without overflow
        self.assertFalse(np.isnan(score.score))
        self.assertFalse(np.isinf(score.score))


class TestCircularStatisticsEdgeCases(unittest.TestCase):
    """Test edge cases for circular statistics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ConstructRegistry()
        self.aggregator = ConstructAggregator(config=AggregationConfig(normalization_method="none", min_features_required=1), construct_registry=self.registry)
    
    def test_circular_midpoint_boundary_values(self):
        """Test circadian midpoint at boundary values (0, 24)."""
        # Test midnight (0 hours)
        result_0 = self.aggregator._apply_circular_transform(0.0)
        self.assertIsInstance(result_0, float)
        self.assertFalse(np.isnan(result_0))
        
        # Test noon (12 hours)
        result_12 = self.aggregator._apply_circular_transform(12.0)
        self.assertIsInstance(result_12, float)
        self.assertFalse(np.isnan(result_12))
        
        # Test near midnight (23.99 hours)
        result_23_99 = self.aggregator._apply_circular_transform(23.99)
        self.assertIsInstance(result_23_99, float)
        self.assertFalse(np.isnan(result_23_99))
    
    def test_circular_sd_edge_cases(self):
        """Test circular standard deviation edge cases."""
        # Empty list
        result_empty = self.aggregator._calculate_circular_sd([])
        self.assertEqual(result_empty, 0.0)
        
        # Single value
        result_single = self.aggregator._calculate_circular_sd([12.0])
        self.assertEqual(result_single, 0.0)
        
        # Identical values
        result_identical = self.aggregator._calculate_circular_sd([12.0, 12.0, 12.0])
        self.assertAlmostEqual(result_identical, 0.0, places=10)
        
        # Opposite values (maximum dispersion)
        result_opposite = self.aggregator._calculate_circular_sd([0.0, 12.0])
        self.assertAlmostEqual(result_opposite, np.pi, places=2)
    
    def test_circular_transform_invalid_values(self):
        """Test circular transform with invalid hour values."""
        # Negative hours
        result_negative = self.aggregator._apply_circular_transform(-1.0)
        self.assertIsInstance(result_negative, float)
        
        # Hours > 24
        result_over_24 = self.aggregator._apply_circular_transform(25.0)
        self.assertIsInstance(result_over_24, float)
        
        # Very large values
        result_large = self.aggregator._apply_circular_transform(1000.0)
        self.assertIsInstance(result_large, float)
        self.assertFalse(np.isnan(result_large))
        self.assertFalse(np.isinf(result_large))


class TestDirectionalTransformsEdgeCases(unittest.TestCase):
    """Test edge cases for directional transformations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ConstructRegistry()
        self.aggregator = ConstructAggregator(config=AggregationConfig(normalization_method="none", min_features_required=1), construct_registry=self.registry)
    
    def test_directional_transform_zero_values(self):
        """Test directional transforms with zero values."""
        feature_values = {"test_feature": 0.0}
        feature_directions = {"test_feature": "negative"}
        feature_aggregations = {"test_feature": "directional_inverse"}
        
        result = self.aggregator._apply_directional_transforms(
            feature_values, feature_directions, feature_aggregations
        )
        
        # Should handle zero without division by zero
        self.assertIn("test_feature", result)
        self.assertFalse(np.isnan(result["test_feature"]))
        self.assertFalse(np.isinf(result["test_feature"]))
    
    def test_directional_transform_very_small_values(self):
        """Test directional transforms with very small values."""
        feature_values = {"test_feature": 1e-15}
        feature_directions = {"test_feature": "negative"}
        feature_aggregations = {"test_feature": "directional_inverse"}
        
        result = self.aggregator._apply_directional_transforms(
            feature_values, feature_directions, feature_aggregations
        )
        
        # Should handle very small values with epsilon stabilization
        self.assertIn("test_feature", result)
        self.assertFalse(np.isnan(result["test_feature"]))
        self.assertFalse(np.isinf(result["test_feature"]))
    
    def test_directional_reverse_undefined_range(self):
        """Test directional reverse with undefined feature range."""
        feature_values = {"test_feature": 0.5}
        feature_directions = {"test_feature": "positive"}
        feature_aggregations = {"test_feature": "directional_reverse"}
        
        result = self.aggregator._apply_directional_transforms(
            feature_values, feature_directions, feature_aggregations
        )
        
        # Should fallback to simple negation when range is undefined
        self.assertIn("test_feature", result)
        self.assertIsInstance(result["test_feature"], float)


class TestNormalizationEdgeCases(unittest.TestCase):
    """Test edge cases for normalization methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ConstructRegistry()
        self.aggregator = ConstructAggregator(config=AggregationConfig(normalization_method="none", min_features_required=1), construct_registry=self.registry)
    
    def test_zscore_no_reference_data(self):
        """Test z-score normalization without reference data."""
        config = AggregationConfig(normalization_method="zscore")
        aggregator = ConstructAggregator(config=config, construct_registry=self.registry)
        
        feature_values = {"test_feature": 1.0}
        
        with self.assertRaises(ValueError):
            aggregator._normalize_features(feature_values, "test_construct")
    
    def test_minmax_no_reference_data(self):
        """Test min-max normalization without reference data."""
        config = AggregationConfig(normalization_method="minmax")
        aggregator = ConstructAggregator(config=config, construct_registry=self.registry)
        
        feature_values = {"test_feature": 1.0}
        
        with self.assertRaises(ValueError):
            aggregator._normalize_features(feature_values, "test_construct")
    
    def test_normalization_zero_variance(self):
        """Test normalization with zero variance reference data."""
        config = AggregationConfig(normalization_method="zscore")
        aggregator = ConstructAggregator(config=config, construct_registry=self.registry)
        
        feature_values = {"test_feature": 1.0}
        reference_data = {"test_feature": [1.0, 1.0, 1.0]}  # Zero variance
        
        result = aggregator._normalize_features(feature_values, "test_construct", reference_data)
        
        # Should handle zero variance gracefully
        self.assertIn("test_feature", result)
        self.assertEqual(result["test_feature"], 0.0)
    
    def test_minmax_identical_values(self):
        """Test min-max normalization with identical min/max values."""
        config = AggregationConfig(normalization_method="minmax")
        aggregator = ConstructAggregator(config=config, construct_registry=self.registry)
        
        feature_values = {"test_feature": 1.0}
        reference_data = {"test_feature": [1.0, 1.0, 1.0]}  # Identical values
        
        result = aggregator._normalize_features(feature_values, "test_construct", reference_data)
        
        # Should handle identical values gracefully
        self.assertIn("test_feature", result)
        self.assertEqual(result["test_feature"], 0.0)


class TestReliabilityEstimationEdgeCases(unittest.TestCase):
    """Test edge cases for reliability estimation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ConstructRegistry()
        self.aggregator = ConstructAggregator(config=AggregationConfig(normalization_method="none", min_features_required=1), construct_registry=self.registry)
    
    def test_reliability_insufficient_features(self):
        """Test reliability estimation with insufficient features."""
        construct_def = self.registry.get_construct("behavioral_activation")
        feature_values = {"activity_volume": 1.0}  # Only one feature
        
        reliability = self.aggregator.calculate_construct_reliability(
            "behavioral_activation", feature_values, construct_def
        )
        
        # Should handle insufficient features gracefully
        self.assertIsNone(reliability['cronbachs_alpha'])
        self.assertEqual(reliability['reliability_type'], 'insufficient_features')
    
    def test_reliability_unknown_model(self):
        """Test reliability estimation with unknown measurement model."""
        # Create a mock construct with unknown model
        from psyconstruct.constructs.registry import ConstructDefinition, FeatureDefinition
        
        feature_def = FeatureDefinition(
            name="test_feature",
            description="Test feature",
            weight=1.0,
            data_type="numeric",
            temporal_granularity="daily",
            unit="count",
            expected_range=(0, 100),
            aggregation="mean",
            validation_status="theoretical",
            construct="test_construct",
            missing_data_strategy="interpolation"
        )
        
        construct_def = ConstructDefinition(
            name="test_construct",
            description="Test construct",
            measurement_model="unknown_model",
            aggregation_type="linear",
            features=[feature_def]
        )
        
        feature_values = {"test_feature": 1.0, "test_feature2": 2.0}
        
        reliability = self.aggregator.calculate_construct_reliability(
            "test_construct", feature_values, construct_def
        )
        
        # Should handle unknown model gracefully
        self.assertIsNone(reliability['cronbachs_alpha'])
        self.assertIsNone(reliability['composite_reliability'])
        self.assertEqual(reliability['reliability_type'], 'not_computable')


class TestAdaptiveQualityEdgeCases(unittest.TestCase):
    """Test edge cases for adaptive quality assessment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.assessor = AdaptiveQualityAssessor()
    
    def test_quality_regime_extreme_values(self):
        """Test quality regime assessment with extreme values."""
        # Perfect data
        perfect_data = {
            'coverage_ratio': 1.0,
            'missing_data_ratio': 0.0,
            'data_quality_score': 1.0,
            'temporal_consistency': 1.0
        }
        regime = self.assessor.assess_data_regime(perfect_data)
        self.assertEqual(regime, QualityRegime.STRICT)
        
        # Terrible data
        terrible_data = {
            'coverage_ratio': 0.0,
            'missing_data_ratio': 1.0,
            'data_quality_score': 0.0,
            'temporal_consistency': 0.0
        }
        regime = self.assessor.assess_data_regime(terrible_data)
        self.assertEqual(regime, QualityRegime.LENIENT)
    
    def test_adaptive_thresholds_boundary_conditions(self):
        """Test adaptive thresholds with boundary conditions."""
        # Test with minimum feature count
        thresholds = self.assessor.calculate_adaptive_thresholds(
            "test_construct", "reflective", 1, QualityRegime.LENIENT
        )
        
        # Should respect absolute minimums
        self.assertGreaterEqual(thresholds['min_features_threshold'], 1)
        self.assertGreaterEqual(thresholds['min_quality_threshold'], 0.3)
        
        # Test with maximum complexity
        thresholds = self.assessor.calculate_adaptive_thresholds(
            "routine_stability", "formative", 10, QualityRegime.STRICT, construct_complexity=1.0
        )
        
        # Should handle high complexity
        self.assertLessEqual(thresholds['min_quality_threshold'], 1.0)
        self.assertGreaterEqual(thresholds['min_features_threshold'], 3)
    
    def test_feature_quality_missing_data(self):
        """Test feature quality evaluation with missing/invalid data."""
        # Empty feature results
        adaptive_thresholds = self.assessor.calculate_adaptive_thresholds(
            "test_construct", "reflective", 2, QualityRegime.MODERATE
        )
        
        quality_assessment = self.assessor.evaluate_feature_quality(
            {}, adaptive_thresholds
        )
        
        # Should handle empty results gracefully
        self.assertEqual(quality_assessment['total_features'], 0)
        self.assertEqual(quality_assessment['high_quality_features'], 0)
        self.assertFalse(quality_assessment['meets_feature_requirement'])
        
        # Feature results without quality information
        feature_results = {
            "feature1": {"value": 1.0},  # No quality field
            "feature2": {"value": 2.0}
        }
        
        quality_assessment = self.assessor.evaluate_feature_quality(
            feature_results, adaptive_thresholds
        )
        
        # Should use default quality for missing quality info
        self.assertEqual(quality_assessment['total_features'], 2)
        self.assertEqual(quality_assessment['high_quality_features'], 2)


class TestHarmonizationEdgeCases(unittest.TestCase):
    """Test edge cases for data harmonization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.harmonizer = DataHarmonizer()
    
    def test_empty_data_arrays(self):
        """Test harmonization with empty data arrays."""
        empty_gps = {
            'timestamp': [],
            'latitude': [],
            'longitude': []
        }
        
        result = self.harmonizer.harmonize_gps_data(empty_gps)
        
        # Should handle empty data gracefully
        self.assertIn('timestamp', result)
        self.assertIn('latitude', result)
        self.assertIn('longitude', result)
        self.assertEqual(len(result['timestamp']), 0)
    
    def test_mismatched_array_lengths(self):
        """Test harmonization with mismatched array lengths."""
        mismatched_gps = {
            'timestamp': [1, 2, 3],
            'latitude': [1.0, 2.0],  # Missing one value
            'longitude': [1.0, 2.0, 3.0]
        }
        
        with self.assertRaises(ValueError):
            self.harmonizer.harmonize_gps_data(mismatched_gps)
    
    def test_invalid_coordinates(self):
        """Test harmonization with invalid GPS coordinates."""
        invalid_gps = {
            'timestamp': [1, 2, 3],
            'latitude': [91.0, -91.0, 181.0],  # Invalid lat/lon values
            'longitude': [1.0, 2.0, 3.0]
        }
        
        # Should handle invalid coordinates gracefully (may warn but not crash)
        result = self.harmonizer.harmonize_gps_data(invalid_gps)
        self.assertIsNotNone(result)
    
    def test_imputation_all_missing(self):
        """Test imputation when all data is missing."""
        all_missing = {
            'timestamp': [1, 2, 3],
            'magnitude': [np.nan, np.nan, np.nan]  # All missing
        }
        
        coverage_analysis = {'coverage_ratio': 0.0, 'max_gap_hours': 24.0}
        
        result = self.harmonizer._apply_imputation(all_missing, 'accelerometer', coverage_analysis)
        
        # Should handle all-missing data gracefully
        self.assertIn('magnitude', result)
        # May still contain NaN values, but shouldn't crash


class TestIntegrationEdgeCases(unittest.TestCase):
    """Integration tests for edge cases across components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ConstructRegistry()
        self.aggregator = ConstructAggregator(config=AggregationConfig(normalization_method="none", min_features_required=1), construct_registry=self.registry)
    
    def test_full_pipeline_minimal_data(self):
        """Test full pipeline with minimal valid data."""
        feature_results = {
            "activity_volume": {
                "weekly_activity_count": 1,
                "quality": 0.6  # Just above minimum
            }
        }
        
        score = self.aggregator.aggregate_construct("behavioral_activation", feature_results)
        
        # Should complete pipeline successfully
        self.assertIsNotNone(score)
        self.assertIsInstance(score.score, float)
        self.assertFalse(np.isnan(score.score))
    
    def test_full_pipeline_noisy_data(self):
        """Test full pipeline with noisy, low-quality data."""
        feature_results = {
            "activity_volume": {
                "weekly_activity_count": 1000,  # Normal value
                "quality": 0.8  # High quality
            },
            "location_diversity": {
                "shannon_entropy": -1.0,  # Invalid entropy
                "quality": 0.2
            },
            "app_usage_breadth": {
                "daily_breadth": np.nan,  # Missing value
                "quality": 0.3
            }
        }
        
        score = self.aggregator.aggregate_construct("behavioral_activation", feature_results)
        
        # Should handle noisy data gracefully
        self.assertIsNotNone(score)
        self.assertFalse(np.isnan(score.score))
        self.assertFalse(np.isinf(score.score))
    
    def test_cross_validation_consistency(self):
        """Test that results are consistent across multiple runs."""
        feature_results = {
            "activity_volume": {
                "weekly_activity_count": 100,
                "quality": 0.8
            },
            "location_diversity": {
                "shannon_entropy": 1.5,
                "quality": 0.9
            }
        }
        
        # Run multiple times
        scores = []
        for _ in range(5):
            score = self.aggregator.aggregate_construct("behavioral_activation", feature_results)
            scores.append(score.score)
        
        # Should be deterministic
        for i in range(1, len(scores)):
            self.assertAlmostEqual(scores[0], scores[i], places=10)


if __name__ == '__main__':
    # Run all edge case tests
    unittest.main(verbosity=2)
