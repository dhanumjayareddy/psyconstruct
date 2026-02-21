"""
Construct registry loader and utilities.

This module provides functionality to load and interact with the construct
registry JSON file, enabling feature-to-construct mapping and validation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeatureDefinition:
    """Definition of a feature within the construct registry."""
    
    name: str
    weight: float
    aggregation: str
    validation_status: str
    description: str
    construct: str
    data_type: str
    temporal_granularity: str
    unit: str
    expected_range: Tuple[Optional[float], Optional[float]]
    missing_data_strategy: str
    direction: Optional[str] = None  # "positive" or "negative"
    range_note: Optional[str] = None  # Additional context for expected range


@dataclass
class ConstructDefinition:
    """Definition of a psychological construct."""
    
    name: str
    description: str
    measurement_model: str
    aggregation_type: str
    features: List[FeatureDefinition]
    
    def get_total_weight(self) -> float:
        """Calculate total weight of all features."""
        return sum(feature.weight for feature in self.features)
    
    def validate_weights(self) -> bool:
        """Validate that feature weights sum to approximately 1.0."""
        total = self.get_total_weight()
        return 0.99 <= total <= 1.01  # Allow small floating point errors
    
    def get_feature(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        for feature in self.features:
            if feature.name == feature_name:
                return feature
        return None


class ConstructRegistry:
    """
    Registry for mapping features to psychological constructs.
    
    This class loads and provides access to the construct registry,
    enabling validation of feature-construct relationships and
    retrieval of construct definitions.
    
    Attributes:
        constructs: Dictionary of construct definitions
        feature_metadata: Dictionary of feature metadata
        aggregation_methods: Dictionary of aggregation method definitions
        validation_status: Dictionary of validation status definitions
        metadata: Registry metadata
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize construct registry.
        
        Args:
            registry_path: Path to registry JSON file. If None, uses default.
            
        Raises:
            FileNotFoundError: If registry file cannot be found
            json.JSONDecodeError: If registry file is malformed
        """
        if registry_path is None:
            registry_path = Path(__file__).parent / "registry.json"
        
        self.registry_path = registry_path
        self._load_registry()
        self._validate_registry()
    
    def _load_registry(self) -> None:
        """Load registry from JSON file."""
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as file:
                self.registry_data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Registry file not found: {self.registry_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error parsing registry file: {e}")
        
        # Extract main sections
        self.constructs = self._parse_constructs()
        self.feature_metadata = self.registry_data.get('feature_metadata', {})
        self.aggregation_methods = self.registry_data.get('aggregation_methods', {})
        self.validation_status = self.registry_data.get('validation_status', {})
        self.metadata = self.registry_data.get('registry_metadata', {})
    
    def _parse_constructs(self) -> Dict[str, ConstructDefinition]:
        """Parse construct definitions from registry data."""
        constructs = {}
        
        for construct_name, construct_data in self.registry_data['constructs'].items():
            features = []
            
            for feature_data in construct_data['features']:
                # Get feature metadata
                feature_meta = self.registry_data['feature_metadata'].get(
                    feature_data['name'], {}
                )
                
                feature = FeatureDefinition(
                    name=feature_data['name'],
                    weight=feature_data['weight'],
                    aggregation=feature_data['aggregation'],
                    validation_status=feature_data['validation_status'],
                    description=feature_data['description'],
                    construct=construct_name,
                    data_type=feature_meta.get('data_type', 'unknown'),
                    temporal_granularity=feature_meta.get('temporal_granularity', 'unknown'),
                    unit=feature_meta.get('unit', 'unknown'),
                    expected_range=self._parse_range(feature_meta.get('expected_range', [None, None])),
                    missing_data_strategy=feature_meta.get('missing_data_strategy', 'unknown'),
                    direction=feature_data.get('direction'),  # New field
                    range_note=feature_meta.get('range_note')  # New field
                )
                features.append(feature)
            
            construct = ConstructDefinition(
                name=construct_name,
                description=construct_data['description'],
                measurement_model=construct_data['measurement_model'],
                aggregation_type=construct_data['aggregation_type'],
                features=features
            )
            constructs[construct_name] = construct
        
        return constructs
    
    def _parse_range(self, range_list: List) -> Tuple[Optional[float], Optional[float]]:
        """Parse expected range from list format."""
        if len(range_list) != 2:
            return (None, None)
        
        # Handle special string values
        def parse_value(val):
            if val is None or val == 'null':
                return None
            elif isinstance(val, str) and val in ['variable', 'unbounded']:
                return None  # Convert to None for practical purposes
            else:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
        
        min_val = parse_value(range_list[0])
        max_val = parse_value(range_list[1])
        
        return (min_val, max_val)
    
    def _validate_registry(self) -> None:
        """Validate registry structure and content."""
        # Validate construct weights
        for construct_name, construct in self.constructs.items():
            if not construct.validate_weights():
                total_weight = construct.get_total_weight()
                raise ValueError(
                    f"Construct '{construct_name}' weights must sum to 1.0, got {total_weight:.3f}"
                )
        
        # Validate feature uniqueness
        feature_names = set()
        for construct in self.constructs.values():
            for feature in construct.features:
                if feature.name in feature_names:
                    raise ValueError(f"Duplicate feature name: {feature.name}")
                feature_names.add(feature.name)
        
        # Validate aggregation methods exist
        for construct in self.constructs.values():
            for feature in construct.features:
                if feature.aggregation not in self.aggregation_methods:
                    raise ValueError(
                        f"Unknown aggregation method '{feature.aggregation}' for feature '{feature.name}'"
                    )
        
        # Validate validation statuses
        valid_statuses = set(self.validation_status.keys())
        for construct in self.constructs.values():
            for feature in construct.features:
                if feature.validation_status not in valid_statuses:
                    raise ValueError(
                        f"Unknown validation status '{feature.validation_status}' for feature '{feature.name}'"
                    )
    
    def get_construct(self, construct_name: str) -> ConstructDefinition:
        """
        Get construct definition by name.
        
        Args:
            construct_name: Name of the construct
            
        Returns:
            Construct definition
            
        Raises:
            KeyError: If construct is not found
        """
        if construct_name not in self.constructs:
            raise KeyError(f"Construct not found: {construct_name}")
        return self.constructs[construct_name]
    
    def get_feature(self, feature_name: str) -> FeatureDefinition:
        """
        Get feature definition by name.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature definition
            
        Raises:
            KeyError: If feature is not found
        """
        for construct in self.constructs.values():
            for feature in construct.features:
                if feature.name == feature_name:
                    return feature
        
        raise KeyError(f"Feature not found: {feature_name}")
    
    def get_features_by_construct(self, construct_name: str) -> List[FeatureDefinition]:
        """
        Get all features belonging to a construct.
        
        Args:
            construct_name: Name of the construct
            
        Returns:
            List of feature definitions
            
        Raises:
            KeyError: If construct is not found
        """
        construct = self.get_construct(construct_name)
        return construct.features
    
    def get_features_by_data_type(self, data_type: str) -> List[FeatureDefinition]:
        """
        Get all features of a specific data type.
        
        Args:
            data_type: Type of sensor data (e.g., 'gps', 'accelerometer')
            
        Returns:
            List of feature definitions
        """
        features = []
        for construct in self.constructs.values():
            for feature in construct.features:
                if feature.data_type == data_type:
                    features.append(feature)
        return features
    
    def get_construct_for_feature(self, feature_name: str) -> Optional['ConstructDefinition']:
        """
        Get the construct definition that a feature belongs to.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            ConstructDefinition object or None if not found
        """
        for construct in self.constructs.values():
            for feature in construct.features:
                if feature.name == feature_name:
                    return construct
        return None
    
    def validate_feature_construct_alignment(self, feature_name: str, construct_name: str) -> bool:
        """
        Validate that a feature belongs to the specified construct.
        
        Args:
            feature_name: Name of the feature
            construct_name: Name of the construct
            
        Returns:
            True if feature belongs to construct, False otherwise
        """
        try:
            actual_construct = self.get_construct_for_feature(feature_name)
            return actual_construct == construct_name
        except KeyError:
            return False
    
    def get_feature_weights(self, construct_name: str) -> Dict[str, float]:
        """
        Get feature weights for a construct.
        
        Args:
            construct_name: Name of the construct
            
        Returns:
            Dictionary mapping feature names to weights
            
        Raises:
            KeyError: If construct is not found
        """
        features = self.get_features_by_construct(construct_name)
        return {feature.name: feature.weight for feature in features}
    
    def get_aggregation_method(self, method_name: str) -> Dict[str, Any]:
        """
        Get aggregation method definition.
        
        Args:
            method_name: Name of the aggregation method
            
        Returns:
            Aggregation method definition
            
        Raises:
            KeyError: If method is not found
        """
        if method_name not in self.aggregation_methods:
            raise KeyError(f"Aggregation method not found: {method_name}")
        return self.aggregation_methods[method_name]
    
    def list_constructs(self) -> List[str]:
        """Get list of all construct names."""
        return list(self.constructs.keys())
    
    def list_features(self) -> List[str]:
        """Get list of all feature names."""
        features = []
        for construct in self.constructs.values():
            features.extend([feature.name for feature in construct.features])
        return sorted(features)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get summary of registry contents.
        
        Returns:
            Dictionary with registry summary statistics
        """
        total_constructs = len(self.constructs)
        total_features = sum(len(construct.features) for construct in self.constructs.values())
        
        # Count by validation status
        validation_counts = {}
        for construct in self.constructs.values():
            for feature in construct.features:
                status = feature.validation_status
                validation_counts[status] = validation_counts.get(status, 0) + 1
        
        # Count by data type
        data_type_counts = {}
        for construct in self.constructs.values():
            for feature in construct.features:
                data_type = feature.data_type
                data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1
        
        return {
            "total_constructs": total_constructs,
            "total_features": total_features,
            "validation_status_counts": validation_counts,
            "data_type_counts": data_type_counts,
            "registry_version": self.metadata.get("version", "unknown"),
            "created_date": self.metadata.get("created_date", "unknown")
        }


# Global registry instance
_global_registry: Optional[ConstructRegistry] = None


def get_registry(registry_path: Optional[Path] = None) -> ConstructRegistry:
    """
    Get global construct registry instance.
    
    Args:
        registry_path: Optional path to custom registry file
        
    Returns:
        Construct registry instance
    """
    global _global_registry
    
    if _global_registry is None or registry_path is not None:
        _global_registry = ConstructRegistry(registry_path)
    
    return _global_registry
