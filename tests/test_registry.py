"""
Unit tests for construct registry module.

Tests registry loading, validation, feature-construct mapping,
and all registry access methods.
"""

import pytest
import tempfile
import json
from pathlib import Path
from psyconstruct.constructs.registry import (
    ConstructRegistry,
    FeatureDefinition,
    ConstructDefinition,
    get_registry
)


class TestFeatureDefinition:
    """Test FeatureDefinition dataclass."""
    
    def test_feature_creation(self):
        """Test creating a feature definition."""
        feature = FeatureDefinition(
            name="test_feature",
            weight=0.25,
            aggregation="mean",
            validation_status="theoretical",
            description="Test feature",
            construct="test_construct",
            data_type="accelerometer",
            temporal_granularity="daily",
            unit="count",
            expected_range=(0.0, 100.0),
            missing_data_strategy="interpolation"
        )
        
        assert feature.name == "test_feature"
        assert feature.weight == 0.25
        assert feature.construct == "test_construct"


class TestConstructDefinition:
    """Test ConstructDefinition dataclass."""
    
    def test_construct_creation(self):
        """Test creating a construct definition."""
        feature = FeatureDefinition(
            name="test_feature",
            weight=1.0,
            aggregation="mean",
            validation_status="theoretical",
            description="Test feature",
            construct="test_construct",
            data_type="accelerometer",
            temporal_granularity="daily",
            unit="count",
            expected_range=(0.0, 100.0),
            missing_data_strategy="interpolation"
        )
        
        construct = ConstructDefinition(
            name="test_construct",
            description="Test construct",
            measurement_model="reflective",
            aggregation_type="linear",
            features=[feature]
        )
        
        assert construct.name == "test_construct"
        assert len(construct.features) == 1
        assert construct.get_total_weight() == 1.0
        assert construct.validate_weights()
    
    def test_weight_validation(self):
        """Test weight validation in construct definition."""
        # Valid weights
        feature = FeatureDefinition(
            name="test_feature", weight=1.0, aggregation="mean", validation_status="theoretical",
            description="Test", construct="test_construct", data_type="test", temporal_granularity="daily",
            unit="test", expected_range=(0.0, 100.0), missing_data_strategy="interpolation"
        )
        feature2 = FeatureDefinition(
            name="feature2", weight=0.0, aggregation="mean", validation_status="theoretical",
            description="Test", construct="test", data_type="test", temporal_granularity="daily",
            unit="test", expected_range=(0, 1), missing_data_strategy="test"
        )
        
        construct = ConstructDefinition(
            name="test", description="test", measurement_model="reflective",
            aggregation_type="linear", features=[feature, feature2]
        )
        assert construct.validate_weights()
        
        # Invalid weights
        feature2.weight = 0.1  # Total = 1.1
        construct.features = [feature, feature2]
        assert not construct.validate_weights()


class TestConstructRegistry:
    """Test ConstructRegistry class."""
    
    def create_test_registry_file(self, registry_data):
        """Create a temporary registry file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(registry_data, f)
            return Path(f.name)
    
    def test_registry_loading(self):
        """Test loading registry from JSON file."""
        registry_data = {
            'constructs': {
                'test_construct': {
                    'description': 'Test construct',
                    'measurement_model': 'reflective',
                    'aggregation_type': 'linear',
                    'features': [
                        {
                            'name': 'test_feature',
                            'weight': 1.0,
                            'aggregation': 'mean',
                            'validation_status': 'theoretical',
                            'description': 'Test feature'
                        }
                    ]
                }
            },
            'feature_metadata': {
                'test_feature': {
                    'data_type': 'accelerometer',
                    'temporal_granularity': 'daily',
                    'unit': 'count',
                    'expected_range': [0, 100],
                    'missing_data_strategy': 'interpolation'
                }
            },
            'aggregation_methods': {
                'mean': {
                    'description': 'Simple mean',
                    'formula': 'mean(x)',
                    'requirements': ['numeric_values']
                }
            },
            'validation_status': {
                'theoretical': {
                    'description': 'Theoretical grounding',
                    'confidence_level': 'low',
                    'requires_validation': True
                }
            },
            'registry_metadata': {
                'version': '1.0.1',
                'created_date': '2026-02-21'
            }
        }
        
        registry_file = self.create_test_registry_file(registry_data)
        
        try:
            registry = ConstructRegistry(registry_file)
            assert len(registry.constructs) == 1
            assert 'test_construct' in registry.constructs
            assert len(registry.constructs['test_construct'].features) == 1
        finally:
            registry_file.unlink()
    
    def test_registry_file_not_found(self):
        """Test error handling when registry file is not found."""
        with pytest.raises(FileNotFoundError):
            ConstructRegistry(Path("nonexistent_file.json"))
    
    def test_invalid_json_file(self):
        """Test error handling for malformed JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": "json": content: [}')
            registry_file = Path(f.name)
        
        try:
            with pytest.raises(Exception):  # Any parsing exception
                ConstructRegistry(registry_file)
        finally:
            registry_file.unlink()
    
    def test_weight_sum_validation(self):
        """Test validation of weight sums in constructs."""
        registry_data = {
            'constructs': {
                'invalid_construct': {
                    'description': 'Invalid construct',
                    'measurement_model': 'reflective',
                    'aggregation_type': 'linear',
                    'features': [
                        {
                            'name': 'feature1',
                            'weight': 0.5,
                            'aggregation': 'mean',
                            'validation_status': 'theoretical',
                            'description': 'Feature 1'
                        },
                        {
                            'name': 'feature2',
                            'weight': 0.6,  # Sum = 1.1 (invalid)
                            'aggregation': 'mean',
                            'validation_status': 'theoretical',
                            'description': 'Feature 2'
                        }
                    ]
                }
            },
            'feature_metadata': {
                'feature1': {'data_type': 'test', 'temporal_granularity': 'daily', 'unit': 'test', 'expected_range': [0, 1], 'missing_data_strategy': 'test'},
                'feature2': {'data_type': 'test', 'temporal_granularity': 'daily', 'unit': 'test', 'expected_range': [0, 1], 'missing_data_strategy': 'test'}
            },
            'aggregation_methods': {'mean': {'description': 'Mean', 'formula': 'mean(x)', 'requirements': ['numeric']}},
            'validation_status': {'theoretical': {'description': 'Theoretical', 'confidence_level': 'low', 'requires_validation': True}},
            'registry_metadata': {'version': '1.0.1'}
        }
        
        registry_file = self.create_test_registry_file(registry_data)
        
        try:
            with pytest.raises(ValueError, match="weights must sum to 1.0"):
                ConstructRegistry(registry_file)
        finally:
            registry_file.unlink()
    
    def test_duplicate_feature_validation(self):
        """Test validation of duplicate feature names."""
        registry_data = {
            'constructs': {
                'construct1': {
                    'description': 'Construct 1',
                    'measurement_model': 'reflective',
                    'aggregation_type': 'linear',
                    'features': [
                        {'name': 'duplicate_feature', 'weight': 1.0, 'aggregation': 'mean', 'validation_status': 'theoretical', 'description': 'Feature'}
                    ]
                },
                'construct2': {
                    'description': 'Construct 2',
                    'measurement_model': 'reflective',
                    'aggregation_type': 'linear',
                    'features': [
                        {'name': 'duplicate_feature', 'weight': 1.0, 'aggregation': 'mean', 'validation_status': 'theoretical', 'description': 'Feature'}
                    ]
                }
            },
            'feature_metadata': {
                'duplicate_feature': {'data_type': 'test', 'temporal_granularity': 'daily', 'unit': 'test', 'expected_range': [0, 1], 'missing_data_strategy': 'test'}
            },
            'aggregation_methods': {'mean': {'description': 'Mean', 'formula': 'mean(x)', 'requirements': ['numeric']}},
            'validation_status': {'theoretical': {'description': 'Theoretical', 'confidence_level': 'low', 'requires_validation': True}},
            'registry_metadata': {'version': '1.0.1'}
        }
        
        registry_file = self.create_test_registry_file(registry_data)
        
        try:
            with pytest.raises(ValueError, match="Duplicate feature name"):
                ConstructRegistry(registry_file)
        finally:
            registry_file.unlink()
    
    def test_get_construct(self):
        """Test getting construct by name."""
        registry = get_registry()  # Use real registry
        construct = registry.get_construct("behavioral_activation")
        assert construct.name == "behavioral_activation"
        assert construct.measurement_model == "reflective"
        assert len(construct.features) == 4
    
    def test_get_construct_not_found(self):
        """Test error when construct is not found."""
        registry = get_registry()
        with pytest.raises(KeyError, match="Construct not found"):
            registry.get_construct("nonexistent_construct")
    
    def test_get_feature(self):
        """Test getting feature by name."""
        registry = get_registry()
        feature = registry.get_feature("activity_volume")
        assert feature.name == "activity_volume"
        assert feature.construct == "behavioral_activation"
        assert feature.weight == 0.25
    
    def test_get_feature_not_found(self):
        """Test error when feature is not found."""
        registry = get_registry()
        with pytest.raises(KeyError, match="Feature not found"):
            registry.get_feature("nonexistent_feature")
    
    def test_get_features_by_construct(self):
        """Test getting features by construct."""
        registry = get_registry()
        features = registry.get_features_by_construct("behavioral_activation")
        assert len(features) == 4
        feature_names = [f.name for f in features]
        assert "activity_volume" in feature_names
        assert "location_diversity" in feature_names
    
    def test_get_features_by_data_type(self):
        """Test getting features by data type."""
        registry = get_registry()
        gps_features = registry.get_features_by_data_type("gps")
        assert len(gps_features) >= 2  # location_diversity, home_confinement, movement_radius
        
        feature_names = [f.name for f in gps_features]
        assert "location_diversity" in feature_names
        assert "home_confinement" in feature_names
    
    def test_get_construct_for_feature(self):
        """Test getting construct for a feature."""
        registry = get_registry()
        construct = registry.get_construct_for_feature("activity_volume")
        assert construct.name == "behavioral_activation"
    
    def test_validate_feature_construct_alignment(self):
        """Test feature-construct alignment validation."""
        registry = get_registry()
        
        # Valid alignment
        assert registry.validate_feature_construct_alignment("activity_volume", "behavioral_activation")
        
        # Invalid alignment
        assert not registry.validate_feature_construct_alignment("activity_volume", "avoidance")
        
        # Nonexistent feature
        assert not registry.validate_feature_construct_alignment("nonexistent", "behavioral_activation")
    
    def test_get_feature_weights(self):
        """Test getting feature weights for a construct."""
        registry = get_registry()
        weights = registry.get_feature_weights("behavioral_activation")
        assert len(weights) == 4
        assert weights["activity_volume"] == 0.25
        assert sum(weights.values()) == 1.0
    
    def test_get_aggregation_method(self):
        """Test getting aggregation method definition."""
        registry = get_registry()
        method = registry.get_aggregation_method("mean")
        assert "description" in method
        assert "formula" in method
        assert "requirements" in method
    
    def test_list_constructs(self):
        """Test listing all constructs."""
        registry = get_registry()
        constructs = registry.list_constructs()
        expected_constructs = ["behavioral_activation", "avoidance", "social_engagement", "routine_stability"]
        assert set(constructs) == set(expected_constructs)
    
    def test_list_features(self):
        """Test listing all features."""
        registry = get_registry()
        features = registry.list_features()
        assert len(features) == 14  # As specified in PRD
        assert "activity_volume" in features
        assert "circadian_midpoint" in features
    
    def test_get_registry_summary(self):
        """Test getting registry summary."""
        registry = get_registry()
        summary = registry.get_registry_summary()
        
        assert summary["total_constructs"] == 4
        assert summary["total_features"] == 14
        assert "validation_status_counts" in summary
        assert "data_type_counts" in summary
        assert "registry_version" in summary
        assert "created_date" in summary
        
        # Check validation status counts
        validation_counts = summary["validation_status_counts"]
        assert "theoretical" in validation_counts
        assert "literature_supported" in validation_counts
        assert "experimental" in validation_counts
        
        # Check data type counts
        data_type_counts = summary["data_type_counts"]
        assert "accelerometer" in data_type_counts
        assert "gps" in data_type_counts
        assert "communication" in data_type_counts
        assert "screen_state" in data_type_counts


class TestGetRegistry:
    """Test get_registry function."""
    
    def test_get_default_registry(self):
        """Test getting default registry instance."""
        registry = get_registry()
        assert isinstance(registry, ConstructRegistry)
        assert len(registry.constructs) == 4
    
    def test_get_custom_registry(self):
        """Test getting custom registry instance."""
        registry_data = {
            'constructs': {
                'custom_construct': {
                    'description': 'Custom construct',
                    'measurement_model': 'reflective',
                    'aggregation_type': 'linear',
                    'features': [
                        {'name': 'custom_feature', 'weight': 1.0, 'aggregation': 'mean', 'validation_status': 'theoretical', 'description': 'Custom feature'}
                    ]
                }
            },
            'feature_metadata': {
                'custom_feature': {'data_type': 'test', 'temporal_granularity': 'daily', 'unit': 'test', 'expected_range': [0, 1], 'missing_data_strategy': 'test'}
            },
            'aggregation_methods': {'mean': {'description': 'Mean', 'formula': 'mean(x)', 'requirements': ['numeric']}},
            'validation_status': {'theoretical': {'description': 'Theoretical', 'confidence_level': 'low', 'requires_validation': True}},
            'registry_metadata': {'version': '1.0.1'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(registry_data, f)
            registry_file = Path(f.name)
        
        try:
            custom_registry = get_registry(registry_file)
            assert len(custom_registry.constructs) == 1
            assert "custom_construct" in custom_registry.constructs
        finally:
            registry_file.unlink()


class TestRegistryIntegration:
    """Integration tests for registry with real data."""
    
    def test_real_registry_structure(self):
        """Test real registry structure and content."""
        # Force fresh registry load to avoid caching issues
        registry = ConstructRegistry()
        
        # Test all constructs exist
        expected_constructs = ["behavioral_activation", "avoidance", "social_engagement", "routine_stability"]
        for construct_name in expected_constructs:
            assert construct_name in registry.constructs
            construct = registry.constructs[construct_name]
            assert construct.validate_weights()
        
        # Test measurement models
        ba_construct = registry.constructs["behavioral_activation"]
        se_construct = registry.constructs["social_engagement"]
        assert ba_construct.measurement_model == "reflective"
        assert se_construct.measurement_model == "formative"
        
        # Test aggregation methods
        for construct in registry.constructs.values():
            for feature in construct.features:
                method = registry.get_aggregation_method(feature.aggregation)
                assert "description" in method
                assert "formula" in method
    
    def test_feature_metadata_completeness(self):
        """Test that all features have complete metadata."""
        registry = get_registry()
        
        for feature_name in registry.list_features():
            feature = registry.get_feature(feature_name)
            
            # Check required fields
            assert feature.data_type != "unknown"
            assert feature.temporal_granularity != "unknown"
            assert feature.unit != "unknown"
            assert feature.missing_data_strategy != "unknown"
            
            # Check expected range is valid
            min_val, max_val = feature.expected_range
            if min_val is not None and max_val is not None:
                assert min_val <= max_val
