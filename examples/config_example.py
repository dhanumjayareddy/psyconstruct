"""
Example usage of psyconstruct configuration module.

This example demonstrates how to:
1. Use default configuration
2. Create custom configuration
3. Get feature-specific parameters
4. Validate configuration parameters
"""

from datetime import timedelta
from psyconstruct.config import (
    PsyconstructConfig,
    DataThresholds,
    ClusteringParameters,
    get_config
)


def example_default_config():
    """Example using default configuration."""
    print("=== Default Configuration Example ===")
    
    # Get default configuration
    config = get_config()
    
    # Print some key parameters
    print(f"Minimum GPS points per day: {config.data_thresholds.min_gps_points_per_day}")
    print(f"Clustering radius: {config.clustering.clustering_radius_meters}m")
    print(f"Sleep minimum duration: {config.sleep.sleep_minimum_duration_hours}h")
    print(f"Daily window: {config.windows.daily_window}")
    
    # Show construct weights
    print("\nBehavioral Activation weights:")
    for feature, weight in config.weights.ba_weights.items():
        print(f"  {feature}: {weight}")
    
    print()


def example_custom_config():
    """Example creating custom configuration."""
    print("=== Custom Configuration Example ===")
    
    # Create custom data thresholds
    custom_thresholds = DataThresholds(
        min_gps_points_per_day=20,  # Require more GPS points
        min_accelerometer_coverage=0.8,  # Require higher coverage
        max_gps_gap_hours=2.0  # Stricter gap tolerance
    )
    
    # Create custom clustering parameters
    custom_clustering = ClusteringParameters(
        clustering_radius_meters=75.0,  # Larger clustering radius
        home_cluster_radius_meters=150.0  # Larger home radius
    )
    
    # Create configuration with custom parameters
    custom_config = PsyconstructConfig(
        data_thresholds=custom_thresholds,
        clustering=custom_clustering
    )
    
    print(f"Custom GPS points per day: {custom_config.data_thresholds.min_gps_points_per_day}")
    print(f"Custom clustering radius: {custom_config.clustering.clustering_radius_meters}m")
    print()


def example_feature_config():
    """Example getting feature-specific configuration."""
    print("=== Feature Configuration Example ===")
    
    config = get_config()
    
    # Get configuration for specific features
    features = ["activity_volume", "location_diversity", "home_confinement", "sleep_onset_consistency"]
    
    for feature in features:
        feature_config = config.get_feature_config(feature)
        print(f"\n{feature.upper()} Configuration:")
        for key, value in feature_config.items():
            print(f"  {key}: {value}")
    
    print()


def example_config_validation():
    """Example configuration validation."""
    print("=== Configuration Validation Example ===")
    
    try:
        # This will raise an error due to invalid weights
        invalid_weights = {
            "activity_volume": 0.5,
            "location_diversity": 0.3,
            "app_usage_breadth": 0.1,
            "activity_timing_variance": 0.2  # Sum = 1.1 (invalid)
        }
        
        from psyconstruct.config import ConstructAggregationWeights
        invalid_config = PsyconstructConfig(
            weights=ConstructAggregationWeights(ba_weights=invalid_weights)
        )
        
    except ValueError as e:
        print(f"Configuration validation error: {e}")
    
    try:
        # This will raise an error due to invalid sleep parameters
        from psyconstruct.config import SleepParameters
        invalid_sleep = SleepParameters(
            sleep_minimum_duration_hours=8.0,
            sleep_maximum_duration_hours=6.0  # Min > Max (invalid)
        )
        
        invalid_config = PsyconstructConfig(sleep=invalid_sleep)
        
    except ValueError as e:
        print(f"Sleep parameter validation error: {e}")
    
    print()


def example_serialization():
    """Example configuration serialization."""
    print("=== Configuration Serialization Example ===")
    
    config = get_config()
    
    # Convert to dictionary
    config_dict = config.to_dict()
    
    print("Configuration serialized to dictionary with keys:")
    for key in config_dict.keys():
        print(f"  {key}")
    
    print(f"\nTotal configuration items: {len(config_dict)}")
    print()


if __name__ == "__main__":
    """Run all examples."""
    print("Psyconstruct Configuration Examples")
    print("=" * 50)
    
    example_default_config()
    example_custom_config()
    example_feature_config()
    example_config_validation()
    example_serialization()
    
    print("All examples completed successfully!")
