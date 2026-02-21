"""
Example usage of construct registry module.

This example demonstrates how to:
1. Load and access the construct registry
2. Query feature-construct relationships
3. Validate feature mappings
4. Extract construct and feature metadata
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psyconstruct.constructs.registry import get_registry


def example_registry_overview():
    """Example showing registry overview and summary."""
    print("=== Registry Overview Example ===")
    
    registry = get_registry()
    summary = registry.get_registry_summary()
    
    print(f"Registry version: {summary['registry_version']}")
    print(f"Created date: {summary['created_date']}")
    print(f"Total constructs: {summary['total_constructs']}")
    print(f"Total features: {summary['total_features']}")
    
    print("\nValidation status distribution:")
    for status, count in summary['validation_status_counts'].items():
        print(f"  {status}: {count} features")
    
    print("\nData type distribution:")
    for data_type, count in summary['data_type_counts'].items():
        print(f"  {data_type}: {count} features")
    
    print()


def example_construct_details():
    """Example showing construct-level details."""
    print("=== Construct Details Example ===")
    
    registry = get_registry()
    
    for construct_name in registry.list_constructs():
        construct = registry.get_construct(construct_name)
        
        print(f"\n{construct_name.upper()}:")
        print(f"  Description: {construct.description}")
        print(f"  Measurement model: {construct.measurement_model}")
        print(f"  Aggregation type: {construct.aggregation_type}")
        print(f"  Total weight: {construct.get_total_weight():.2f}")
        print(f"  Features ({len(construct.features)}):")
        
        for feature in construct.features:
            print(f"    - {feature.name} (weight: {feature.weight}, aggregation: {feature.aggregation})")
    
    print()


def example_feature_details():
    """Example showing feature-level details."""
    print("=== Feature Details Example ===")
    
    registry = get_registry()
    
    # Show details for a few representative features
    sample_features = ["activity_volume", "home_confinement", "communication_frequency", "sleep_onset_consistency"]
    
    for feature_name in sample_features:
        feature = registry.get_feature(feature_name)
        
        print(f"\n{feature_name.upper()}:")
        print(f"  Construct: {feature.construct}")
        print(f"  Description: {feature.description}")
        print(f"  Data type: {feature.data_type}")
        print(f"  Temporal granularity: {feature.temporal_granularity}")
        print(f"  Unit: {feature.unit}")
        print(f"  Expected range: {feature.expected_range}")
        print(f"  Missing data strategy: {feature.missing_data_strategy}")
        print(f"  Validation status: {feature.validation_status}")
        print(f"  Weight: {feature.weight}")
        print(f"  Aggregation: {feature.aggregation}")
    
    print()


def example_feature_queries():
    """Example showing various feature queries."""
    print("=== Feature Queries Example ===")
    
    registry = get_registry()
    
    # Get features by construct
    print("Features in Behavioral Activation:")
    ba_features = registry.get_features_by_construct("behavioral_activation")
    for feature in ba_features:
        print(f"  - {feature.name}")
    
    # Get features by data type
    print("\nGPS-based features:")
    gps_features = registry.get_features_by_data_type("gps")
    for feature in gps_features:
        print(f"  - {feature.name} (in {feature.construct})")
    
    # Get construct for feature
    print(f"\nConstruct for 'communication_frequency': {registry.get_construct_for_feature('communication_frequency')}")
    
    # Get feature weights
    print("\nAvoidance construct weights:")
    av_weights = registry.get_feature_weights("avoidance")
    for feature, weight in av_weights.items():
        print(f"  {feature}: {weight}")
    
    print()


def example_aggregation_methods():
    """Example showing aggregation method details."""
    print("=== Aggregation Methods Example ===")
    
    registry = get_registry()
    
    # Show all aggregation methods
    print("Available aggregation methods:")
    for method_name in registry.aggregation_methods.keys():
        method = registry.get_aggregation_method(method_name)
        print(f"\n{method_name.upper()}:")
        print(f"  Description: {method['description']}")
        print(f"  Formula: {method['formula']}")
        print(f"  Requirements: {', '.join(method['requirements'])}")
    
    # Show which features use each method
    print("\nFeature usage by aggregation method:")
    method_usage = {}
    for construct in registry.constructs.values():
        for feature in construct.features:
            method = feature.aggregation
            if method not in method_usage:
                method_usage[method] = []
            method_usage[method].append(feature.name)
    
    for method, features in method_usage.items():
        print(f"  {method}: {', '.join(features)}")
    
    print()


def example_validation_operations():
    """Example showing validation operations."""
    print("=== Validation Operations Example ===")
    
    registry = get_registry()
    
    # Test feature-construct alignment
    test_cases = [
        ("activity_volume", "behavioral_activation"),  # Valid
        ("activity_volume", "avoidance"),              # Invalid
        ("nonexistent_feature", "behavioral_activation")  # Nonexistent
    ]
    
    for feature, construct in test_cases:
        is_valid = registry.validate_feature_construct_alignment(feature, construct)
        status = "✓" if is_valid else "✗"
        print(f"{status} {feature} -> {construct}")
    
    # Show validation status definitions
    print("\nValidation status definitions:")
    for status, definition in registry.validation_status.items():
        print(f"\n{status.upper()}:")
        print(f"  Description: {definition['description']}")
        print(f"  Confidence level: {definition['confidence_level']}")
        print(f"  Requires validation: {definition['requires_validation']}")
    
    print()


def example_measurement_models():
    """Example showing measurement model information."""
    print("=== Measurement Models Example ===")
    
    registry = get_registry()
    
    # Group constructs by measurement model
    reflective_constructs = []
    formative_constructs = []
    
    for construct_name, construct in registry.constructs.items():
        if construct.measurement_model == "reflective":
            reflective_constructs.append(construct_name)
        elif construct.measurement_model == "formative":
            formative_constructs.append(construct_name)
    
    print("Reflective measurement model constructs:")
    for construct in reflective_constructs:
        print(f"  - {construct}")
    
    print("\nFormative measurement model constructs:")
    for construct in formative_constructs:
        print(f"  - {construct}")
    
    # Show theoretical references
    print("\nTheoretical references from registry metadata:")
    if 'theoretical_basis' in registry.metadata:
        for construct, reference in registry.metadata['theoretical_basis'].items():
            print(f"  {construct}: {reference}")
    
    if 'measurement_references' in registry.metadata:
        print("\nMeasurement model references:")
        for model, reference in registry.metadata['measurement_references'].items():
            print(f"  {model}: {reference}")
    
    print()


def example_registry_search():
    """Example showing registry search capabilities."""
    print("=== Registry Search Example ===")
    
    registry = get_registry()
    
    # Find features by validation status
    print("Validated features:")
    all_features = registry.list_features()
    validated_features = [
        feature for feature in all_features
        if registry.get_feature(feature).validation_status == "validated"
    ]
    for feature in validated_features:
        print(f"  - {feature}")
    
    # Find experimental features
    print("\nExperimental features:")
    experimental_features = [
        feature for feature in all_features
        if registry.get_feature(feature).validation_status == "experimental"
    ]
    for feature in experimental_features:
        print(f"  - {feature}")
    
    # Find daily granularity features
    print("\nDaily granularity features:")
    daily_features = [
        feature for feature in all_features
        if registry.get_feature(feature).temporal_granularity == "daily"
    ]
    for feature in daily_features:
        print(f"  - {feature}")
    
    print()


if __name__ == "__main__":
    """Run all examples."""
    print("Psyconstruct Registry Examples")
    print("=" * 50)
    
    example_registry_overview()
    example_construct_details()
    example_feature_details()
    example_feature_queries()
    example_aggregation_methods()
    example_validation_operations()
    example_measurement_models()
    example_registry_search()
    
    print("All registry examples completed successfully!")
