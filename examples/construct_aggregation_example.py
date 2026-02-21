"""
Example usage of Construct Aggregator for Psyconstruct.

This example demonstrates how to:
1. Aggregate individual features into construct-level scores
2. Apply different normalization methods
3. Handle missing data and quality issues
4. Export and interpret construct scores
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psyconstruct.constructs import (
    ConstructAggregator,
    AggregationConfig,
    ConstructScore
)


def create_mock_feature_results():
    """Create mock feature extraction results for demonstration."""
    
    print("Creating mock feature extraction results...")
    
    # Behavioral Activation features
    ba_features = {
        "activity_volume": {
            "weekly_activity_count": 1250.5,
            "quality_metrics": {"overall_quality": 0.85}
        },
        "location_diversity": {
            "shannon_entropy": 2.3,
            "quality_metrics": {"overall_quality": 0.78}
        },
        "app_usage_breadth": {
            "daily_breadth": 4.2,
            "quality_metrics": {"overall_quality": 0.92}
        },
        "activity_timing_variance": {
            "timing_variance": 0.15,
            "quality_metrics": {"overall_quality": 0.81}
        }
    }
    
    # Avoidance features
    avoidance_features = {
        "home_confinement": {
            "home_confinement_percentage": 65.3,
            "quality_metrics": {"overall_quality": 0.88}
        },
        "communication_gaps": {
            "max_daily_gap_hours": 8.5,
            "quality_metrics": {"overall_quality": 0.75}
        },
        "movement_radius": {
            "radius_of_gyration_meters": 1250.0,
            "quality_metrics": {"overall_quality": 0.83}
        }
    }
    
    # Social Engagement features
    se_features = {
        "communication_frequency": {
            "weekly_outgoing_count": 45.0,
            "quality_metrics": {"overall_quality": 0.90}
        },
        "contact_diversity": {
            "weekly_diversity": 12.0,
            "quality_metrics": {"overall_quality": 0.85}
        },
        "initiation_rate": {
            "weekly_initiation_rate": 0.65,
            "quality_metrics": {"overall_quality": 0.82}
        }
    }
    
    # Routine Stability features
    rs_features = {
        "sleep_onset_consistency": {
            "sleep_onset_sd_hours": 1.2,
            "quality_metrics": {"overall_quality": 0.79}
        },
        "sleep_duration": {
            "mean_sleep_duration_hours": 7.8,
            "quality_metrics": {"overall_quality": 0.91}
        },
        "activity_fragmentation": {
            "mean_entropy": 2.1,
            "quality_metrics": {"overall_quality": 0.77}
        },
        "circadian_midpoint": {
            "mean_midpoint_hour": 3.5,
            "quality_metrics": {"overall_quality": 0.84}
        }
    }
    
    # Combine all features
    all_features = {}
    all_features.update(ba_features)
    all_features.update(avoidance_features)
    all_features.update(se_features)
    all_features.update(rs_features)
    
    return all_features


def create_reference_data():
    """Create mock reference data for normalization."""
    
    print("Creating reference data for normalization...")
    
    reference_data = {
        "behavioral_activation": {
            "activity_volume": [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
            "location_diversity": [1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5],
            "app_usage_breadth": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            "activity_timing_variance": [0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28]
        },
        "avoidance": {
            "home_confinement": [20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
            "communication_gaps": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "movement_radius": [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000]
        },
        "social_engagement": {
            "communication_frequency": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "contact_diversity": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            "initiation_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
        "routine_stability": {
            "sleep_onset_consistency": [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2],
            "sleep_duration": [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5],
            "activity_fragmentation": [1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2],
            "circadian_midpoint": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
        }
    }
    
    return reference_data


def example_basic_aggregation():
    """Example showing basic construct aggregation."""
    print("=== Basic Construct Aggregation Example ===")
    
    # Initialize aggregator with default configuration
    config = AggregationConfig(
        normalization_method="zscore",
        aggregation_method="weighted_mean",
        min_features_required=2
    )
    
    aggregator = ConstructAggregator(config=config)
    
    # Create mock feature results
    feature_results = create_mock_feature_results()
    
    # Aggregate individual constructs
    constructs = ["behavioral_activation", "avoidance", "social_engagement", "routine_stability"]
    
    for construct_name in constructs:
        try:
            score = aggregator.aggregate_construct(
                construct_name, 
                feature_results, 
                participant_id="demo_participant"
            )
            
            print(f"\n{construct_name.title()}:")
            print(f"  Raw Score: {score.score:.3f}")
            print(f"  Normalized Score: {score.normalized_score:.3f}")
            print(f"  Overall Quality: {score.quality_metrics['overall_quality']:.3f}")
            print(f"  Features Used: {list(score.feature_scores.keys())}")
            print(f"  Interpretation: {score.interpretation}")
            
            if score.confidence_interval:
                print(f"  95% CI: [{score.confidence_interval[0]:.3f}, {score.confidence_interval[1]:.3f}]")
            
        except ValueError as e:
            print(f"\n{construct_name.title()}: Failed - {str(e)}")
    
    print()


def example_normalization_methods():
    """Example showing different normalization methods."""
    print("=== Normalization Methods Comparison Example ===")
    
    feature_results = create_mock_feature_results()
    reference_data = create_reference_data()
    
    # Test different normalization methods
    methods = ["none", "zscore", "minmax", "robust"]
    
    for method in methods:
        print(f"\nNormalization Method: {method}")
        print("-" * 40)
        
        config = AggregationConfig(
            normalization_method=method,
            aggregation_method="weighted_mean"
        )
        
        aggregator = ConstructAggregator(config=config)
        
        try:
            # Aggregate behavioral activation as example
            score = aggregator.aggregate_construct(
                "behavioral_activation",
                feature_results,
                participant_id="demo_participant",
                reference_data=reference_data["behavioral_activation"]
            )
            
            print(f"  Raw Score: {score.score:.3f}")
            print(f"  Normalized Score: {score.normalized_score:.3f}")
            
            # Show individual feature scores
            print(f"  Feature Scores:")
            for feature, value in score.feature_scores.items():
                print(f"    {feature}: {value:.3f}")
                
        except ValueError as e:
            print(f"  Error: {str(e)}")
    
    print()


def example_aggregation_methods():
    """Example showing different aggregation methods."""
    print("=== Aggregation Methods Comparison Example ===")
    
    feature_results = create_mock_feature_results()
    reference_data = create_reference_data()
    
    # Test different aggregation methods
    methods = ["weighted_mean", "unweighted_mean", "median"]
    
    for method in methods:
        print(f"\nAggregation Method: {method}")
        print("-" * 40)
        
        config = AggregationConfig(
            normalization_method="zscore",
            aggregation_method=method
        )
        
        aggregator = ConstructAggregator(config=config)
        
        try:
            # Aggregate social engagement as example
            score = aggregator.aggregate_construct(
                "social_engagement",
                feature_results,
                participant_id="demo_participant",
                reference_data=reference_data["social_engagement"]
            )
            
            print(f"  Raw Score: {score.score:.3f}")
            print(f"  Normalized Score: {score.normalized_score:.3f}")
            print(f"  Quality Metrics: {score.quality_metrics}")
            
        except ValueError as e:
            print(f"  Error: {str(e)}")
    
    print()


def example_quality_handling():
    """Example showing quality-based feature filtering."""
    print("=== Quality Handling Example ===")
    
    # Create feature results with varying quality
    feature_results = create_mock_feature_results()
    
    # Simulate poor quality for some features
    feature_results["activity_volume"]["quality_metrics"]["overall_quality"] = 0.3  # Low quality
    feature_results["location_diversity"]["quality_metrics"]["overall_quality"] = 0.4  # Low quality
    
    print("Feature Qualities:")
    for feature, result in feature_results.items():
        quality = result["quality_metrics"]["overall_quality"]
        print(f"  {feature}: {quality:.2f}")
    
    # Test with different quality thresholds
    thresholds = [0.2, 0.5, 0.8]
    
    for threshold in thresholds:
        print(f"\nQuality Threshold: {threshold}")
        print("-" * 30)
        
        config = AggregationConfig(
            min_quality_threshold=threshold,
            min_features_required=2
        )
        
        aggregator = ConstructAggregator(config=config)
        
        try:
            score = aggregator.aggregate_construct(
                "behavioral_activation",
                feature_results,
                participant_id="demo_participant"
            )
            
            print(f"  Success! Score: {score.normalized_score:.3f}")
            print(f"  Features Used: {list(score.feature_scores.keys())}")
            print(f"  Quality Score: {score.quality_metrics['overall_quality']:.3f}")
            
        except ValueError as e:
            print(f"  Failed: {str(e)}")
    
    print()


def example_missing_data_handling():
    """Example showing missing data handling."""
    print("=== Missing Data Handling Example ===")
    
    # Create feature results with missing features
    feature_results = create_mock_feature_results()
    
    # Remove some features to simulate missing data
    missing_features = ["app_usage_breadth", "communication_gaps", "sleep_duration"]
    
    print(f"Simulating missing features: {missing_features}")
    
    for feature in missing_features:
        if feature in feature_results:
            del feature_results[feature]
    
    print(f"Available features: {list(feature_results.keys())}")
    
    # Test different minimum feature requirements
    min_features_list = [1, 2, 3]
    
    for min_features in min_features_list:
        print(f"\nMinimum Features Required: {min_features}")
        print("-" * 40)
        
        config = AggregationConfig(
            min_features_required=min_features,
            handle_missing="exclude"
        )
        
        aggregator = ConstructAggregator(config=config)
        
        # Test each construct
        for construct_name in ["behavioral_activation", "avoidance", "social_engagement", "routine_stability"]:
            try:
                score = aggregator.aggregate_construct(
                    construct_name,
                    feature_results,
                    participant_id="demo_participant"
                )
                
                print(f"  {construct_name}: {score.normalized_score:.3f} ({len(score.feature_scores)} features)")
                
            except ValueError as e:
                print(f"  {construct_name}: Failed - {str(e)}")
    
    print()


def example_batch_aggregation():
    """Example showing batch aggregation of all constructs."""
    print("=== Batch Aggregation Example ===")
    
    # Initialize aggregator
    config = AggregationConfig(
        normalization_method="zscore",
        aggregation_method="weighted_mean",
        include_feature_scores=True,
        include_quality_metrics=True
    )
    
    aggregator = ConstructAggregator(config=config)
    
    # Create feature results and reference data
    feature_results = create_mock_feature_results()
    reference_data = create_reference_data()
    
    # Aggregate all constructs
    print("Aggregating all constructs...")
    
    construct_scores = aggregator.aggregate_all_constructs(
        feature_results,
        participant_id="demo_participant",
        reference_data=reference_data
    )
    
    print(f"\nSuccessfully aggregated {len(construct_scores)} constructs:")
    
    # Display results
    for construct_name, score in construct_scores.items():
        print(f"\n{construct_name.title()}:")
        print(f"  Score: {score.normalized_score:.3f}")
        print(f"  Quality: {score.quality_metrics['overall_quality']:.3f}")
        print(f"  Features: {len(score.feature_scores)}")
        print(f"  Interpretation: {score.interpretation}")
    
    # Export results
    print(f"\nExporting results...")
    
    # Export as JSON
    aggregator.export_scores(
        construct_scores,
        "construct_scores.json",
        format="json"
    )
    print("  Exported to construct_scores.json")
    
    # Export as CSV
    aggregator.export_scores(
        construct_scores,
        "construct_scores.csv",
        format="csv"
    )
    print("  Exported to construct_scores.csv")
    
    print()


def example_clinical_interpretation():
    """Example showing clinical interpretation of construct scores."""
    print("=== Clinical Interpretation Example ===")
    
    # Create different clinical profiles
    profiles = {
        "Healthy Baseline": {
            "activity_volume": {"weekly_activity_count": 1200, "quality_metrics": {"overall_quality": 0.9}},
            "location_diversity": {"shannon_entropy": 2.5, "quality_metrics": {"overall_quality": 0.85}},
            "app_usage_breadth": {"daily_breadth": 4.0, "quality_metrics": {"overall_quality": 0.9}},
            "activity_timing_variance": {"timing_variance": 0.15, "quality_metrics": {"overall_quality": 0.8}},
            "home_confinement": {"home_confinement_percentage": 30, "quality_metrics": {"overall_quality": 0.85}},
            "communication_gaps": {"max_daily_gap_hours": 3, "quality_metrics": {"overall_quality": 0.8}},
            "movement_radius": {"radius_of_gyration_meters": 2000, "quality_metrics": {"overall_quality": 0.9}},
            "communication_frequency": {"weekly_outgoing_count": 60, "quality_metrics": {"overall_quality": 0.9}},
            "contact_diversity": {"weekly_diversity": 15, "quality_metrics": {"overall_quality": 0.85}},
            "initiation_rate": {"weekly_initiation_rate": 0.7, "quality_metrics": {"overall_quality": 0.8}},
            "sleep_onset_consistency": {"sleep_onset_sd_hours": 0.8, "quality_metrics": {"overall_quality": 0.9}},
            "sleep_duration": {"mean_sleep_duration_hours": 8.0, "quality_metrics": {"overall_quality": 0.95}},
            "activity_fragmentation": {"mean_entropy": 2.0, "quality_metrics": {"overall_quality": 0.8}},
            "circadian_midpoint": {"mean_midpoint_hour": 3.0, "quality_metrics": {"overall_quality": 0.85}}
        },
        "Depressive Profile": {
            "activity_volume": {"weekly_activity_count": 600, "quality_metrics": {"overall_quality": 0.8}},
            "location_diversity": {"shannon_entropy": 1.5, "quality_metrics": {"overall_quality": 0.75}},
            "app_usage_breadth": {"daily_breadth": 2.0, "quality_metrics": {"overall_quality": 0.8}},
            "activity_timing_variance": {"timing_variance": 0.25, "quality_metrics": {"overall_quality": 0.7}},
            "home_confinement": {"home_confinement_percentage": 80, "quality_metrics": {"overall_quality": 0.9}},
            "communication_gaps": {"max_daily_gap_hours": 12, "quality_metrics": {"overall_quality": 0.85}},
            "movement_radius": {"radius_of_gyration_meters": 800, "quality_metrics": {"overall_quality": 0.8}},
            "communication_frequency": {"weekly_outgoing_count": 20, "quality_metrics": {"overall_quality": 0.75}},
            "contact_diversity": {"weekly_diversity": 5, "quality_metrics": {"overall_quality": 0.7}},
            "initiation_rate": {"weekly_initiation_rate": 0.3, "quality_metrics": {"overall_quality": 0.75}},
            "sleep_onset_consistency": {"sleep_onset_sd_hours": 2.5, "quality_metrics": {"overall_quality": 0.8}},
            "sleep_duration": {"mean_sleep_duration_hours": 6.5, "quality_metrics": {"overall_quality": 0.85}},
            "activity_fragmentation": {"mean_entropy": 3.0, "quality_metrics": {"overall_quality": 0.75}},
            "circadian_midpoint": {"mean_midpoint_hour": 4.5, "quality_metrics": {"overall_quality": 0.8}}
        },
        "Anxiety Profile": {
            "activity_volume": {"weekly_activity_count": 1400, "quality_metrics": {"overall_quality": 0.85}},
            "location_diversity": {"shannon_entropy": 2.0, "quality_metrics": {"overall_quality": 0.8}},
            "app_usage_breadth": {"daily_breadth": 3.5, "quality_metrics": {"overall_quality": 0.85}},
            "activity_timing_variance": {"timing_variance": 0.3, "quality_metrics": {"overall_quality": 0.75}},
            "home_confinement": {"home_confinement_percentage": 50, "quality_metrics": {"overall_quality": 0.8}},
            "communication_gaps": {"max_daily_gap_hours": 6, "quality_metrics": {"overall_quality": 0.8}},
            "movement_radius": {"radius_of_gyration_meters": 1500, "quality_metrics": {"overall_quality": 0.85}},
            "communication_frequency": {"weekly_outgoing_count": 80, "quality_metrics": {"overall_quality": 0.9}},
            "contact_diversity": {"weekly_diversity": 10, "quality_metrics": {"overall_quality": 0.8}},
            "initiation_rate": {"weekly_initiation_rate": 0.8, "quality_metrics": {"overall_quality": 0.85}},
            "sleep_onset_consistency": {"sleep_onset_sd_hours": 1.8, "quality_metrics": {"overall_quality": 0.8}},
            "sleep_duration": {"mean_sleep_duration_hours": 7.0, "quality_metrics": {"overall_quality": 0.8}},
            "activity_fragmentation": {"mean_entropy": 2.8, "quality_metrics": {"overall_quality": 0.75}},
            "circadian_midpoint": {"mean_midpoint_hour": 2.5, "quality_metrics": {"overall_quality": 0.8}}
        }
    }
    
    # Initialize aggregator
    config = AggregationConfig(
        normalization_method="zscore",
        aggregation_method="weighted_mean"
    )
    
    aggregator = ConstructAggregator(config=config)
    
    # Analyze each profile
    for profile_name, feature_results in profiles.items():
        print(f"\n{profile_name}:")
        print("-" * 50)
        
        try:
            construct_scores = aggregator.aggregate_all_constructs(
                feature_results,
                participant_id=f"demo_{profile_name.lower().replace(' ', '_')}"
            )
            
            # Clinical interpretation
            print(f"Construct Scores:")
            for construct_name, score in construct_scores.items():
                print(f"  {construct_name}: {score.normalized_score:.2f} - {score.interpretation}")
            
            # Risk assessment
            risks = []
            if construct_scores["behavioral_activation"].normalized_score < -0.5:
                risks.append("Reduced behavioral activation")
            if construct_scores["avoidance"].normalized_score > 0.5:
                risks.append("High avoidance behaviors")
            if construct_scores["social_engagement"].normalized_score < -0.5:
                risks.append("Social withdrawal")
            if construct_scores["routine_stability"].normalized_score < -0.5:
                risks.append("Routine disruption")
            
            if risks:
                print(f"Clinical Risks: {', '.join(risks)}")
            else:
                print(f"Clinical Risks: No significant risks detected")
            
            # Treatment recommendations
            if profile_name == "Depressive Profile":
                print(f"Treatment Focus: Behavioral activation, social engagement, routine stabilization")
            elif profile_name == "Anxiety Profile":
                print(f"Treatment Focus: Routine stability, exposure therapy, relaxation techniques")
            else:
                print(f"Treatment Focus: Maintain healthy patterns, monitor for changes")
                
        except Exception as e:
            print(f"Error analyzing profile: {str(e)}")
    
    print()


def example_research_configuration():
    """Example showing research-grade configuration."""
    print("=== Research Configuration Example ===")
    
    # Research-grade configuration
    research_config = AggregationConfig(
        normalization_method="zscore",
        within_participant=True,
        aggregation_method="weighted_mean",
        handle_missing="exclude",
        min_features_required=3,
        min_quality_threshold=0.7,
        include_feature_scores=True,
        include_quality_metrics=True,
        include_normalization_params=True
    )
    
    print("Research Configuration:")
    print(f"  Normalization: {research_config.normalization_method}")
    print(f"  Within-participant: {research_config.within_participant}")
    print(f"  Aggregation: {research_config.aggregation_method}")
    print(f"  Min features: {research_config.min_features_required}")
    print(f"  Quality threshold: {research_config.min_quality_threshold}")
    print(f"  Include feature scores: {research_config.include_feature_scores}")
    print(f"  Include quality metrics: {research_config.include_quality_metrics}")
    
    # Initialize with research config
    aggregator = ConstructAggregator(config=research_config)
    
    # Create high-quality research data
    feature_results = create_mock_feature_results()
    reference_data = create_reference_data()
    
    print(f"\nAnalyzing with research configuration...")
    
    try:
        construct_scores = aggregator.aggregate_all_constructs(
            feature_results,
            participant_id="research_participant_001",
            reference_data=reference_data
        )
        
        print(f"Successfully analyzed {len(construct_scores)} constructs")
        
        # Show detailed results for one construct
        ba_score = construct_scores["behavioral_activation"]
        print(f"\nDetailed Results - Behavioral Activation:")
        print(f"  Raw Score: {ba_score.score:.4f}")
        print(f"  Normalized Score: {ba_score.normalized_score:.4f}")
        print(f"  Confidence Interval: {ba_score.confidence_interval}")
        print(f"  Feature Scores: {ba_score.feature_scores}")
        print(f"  Quality Metrics: {ba_score.quality_metrics}")
        print(f"  Aggregation Parameters: {ba_score.aggregation_parameters}")
        print(f"  Timestamp: {ba_score.timestamp}")
        
    except Exception as e:
        print(f"Error with research configuration: {str(e)}")
    
    print()


if __name__ == "__main__":
    """Run all construct aggregation examples."""
    print("Psyconstruct Construct Aggregator Examples")
    print("=" * 50)
    
    example_basic_aggregation()
    example_normalization_methods()
    example_aggregation_methods()
    example_quality_handling()
    example_missing_data_handling()
    example_batch_aggregation()
    example_clinical_interpretation()
    example_research_configuration()
    
    print("All construct aggregation examples completed successfully!")
