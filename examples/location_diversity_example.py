"""
Example usage of location diversity feature.

This example demonstrates how to:
1. Extract location diversity from GPS data
2. Configure clustering and entropy parameters
3. Interpret location diversity results
4. Handle different GPS data quality scenarios
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psyconstruct.features.behavioral_activation import (
    BehavioralActivationFeatures,
    LocationDiversityConfig
)


def create_realistic_gps_data(days: int = 7, points_per_day: int = 50):
    """Create realistic GPS data with daily patterns."""
    
    print(f"Generating {days} days of GPS data with {points_per_day} points per day...")
    
    timestamps = []
    latitudes = []
    longitudes = []
    
    # Define location clusters (NYC area)
    locations = {
        'home': (40.7128, -74.0060),      # City Hall area
        'work': (40.7580, -73.9855),      # Times Square area  
        'gym': (40.7829, -73.9654),       # Central Park area
        'cafe': (40.7489, -73.9680),      # Grand Central area
        'grocery': (40.7282, -73.9942),   # Chelsea area
        'park': (40.7614, -73.9776)       # Upper East Side
    }
    
    base_time = datetime(2026, 2, 21, 6, 0, 0)  # Start at 6 AM
    
    for day in range(days):
        current_day_offset = day * 24 * 60  # minutes
        
        for point in range(points_per_day):
            # Distribute points throughout the day
            minute_of_day = int(point * 24 * 60 / points_per_day)
            total_minute = current_day_offset + minute_of_day
            hour = (total_minute % (24 * 60)) / 60
            
            # Simulate daily movement patterns
            if 6 <= hour < 8:      # Morning routine at home
                location = 'home'
            elif 8 <= hour < 8.5:   # Commute to work
                location = random.choice(['cafe', 'grocery'])
            elif 8.5 <= hour < 12:  # Work morning
                location = 'work'
            elif 12 <= hour < 13:   # Lunch break
                location = random.choice(['cafe', 'grocery'])
            elif 13 <= hour < 17:   # Work afternoon
                location = 'work'
            elif 17 <= hour < 18:   # Commute home
                location = random.choice(['gym', 'park'])
            elif 18 <= hour < 20:   # Evening activities
                location = random.choice(['gym', 'park', 'cafe'])
            elif 20 <= hour < 23:   # Evening at home
                location = 'home'
            else:                    # Night at home
                location = 'home'
            
            # Get base coordinates
            base_lat, base_lon = locations[location]
            
            # Add realistic GPS noise (±5-10 meters)
            lat_noise = (random.random() - 0.5) * 0.0001  # ~10 meters
            lon_noise = (random.random() - 0.5) * 0.0001
            
            lat = base_lat + lat_noise
            lon = base_lon + lon_noise
            
            # Create timestamp
            timestamp = base_time + timedelta(minutes=total_minute)
            
            timestamps.append(timestamp)
            latitudes.append(lat)
            longitudes.append(lon)
    
    return {
        'timestamp': timestamps,
        'latitude': latitudes,
        'longitude': longitudes
    }


def create_high_diversity_data():
    """Create GPS data with high location diversity."""
    
    print("Creating high diversity GPS data (many different locations)...")
    
    # Many diverse locations around NYC
    diverse_locations = [
        (40.7128, -74.0060), (40.7580, -73.9855), (40.7489, -73.9680),
        (40.7829, -73.9654), (40.7282, -73.9942), (40.7614, -73.9776),
        (40.6892, -74.0445), (40.7282, -73.7949), (40.6782, -73.9442),
        (40.7489, -73.9857), (40.7527, -73.9772), (40.7061, -74.0087)
    ]
    
    timestamps = []
    latitudes = []
    longitudes = []
    
    base_time = datetime(2026, 2, 21, 8, 0, 0)
    
    for day in range(7):
        for point in range(30):  # 30 points per day
            # Visit many different locations
            location_idx = random.randint(0, len(diverse_locations) - 1)
            base_lat, base_lon = diverse_locations[location_idx]
            
            # Add GPS noise
            lat = base_lat + (random.random() - 0.5) * 0.0001
            lon = base_lon + (random.random() - 0.5) * 0.0001
            
            timestamp = base_time + timedelta(days=day, hours=8 + point * 16/30)
            
            timestamps.append(timestamp)
            latitudes.append(lat)
            longitudes.append(lon)
    
    return {
        'timestamp': timestamps,
        'latitude': latitudes,
        'longitude': longitudes
    }


def create_low_diversity_data():
    """Create GPS data with low location diversity (mostly home/work)."""
    
    print("Creating low diversity GPS data (mostly home and work)...")
    
    timestamps = []
    latitudes = []
    longitudes = []
    
    # Only two main locations
    home = (40.7128, -74.0060)
    work = (40.7580, -73.9855)
    
    base_time = datetime(2026, 2, 21, 6, 0, 0)
    
    for day in range(7):
        for point in range(40):  # 40 points per day
            hour = 6 + point * 18 / 40  # 6 AM to midnight
            
            # Simple pattern: home in morning/evening, work during day
            if hour < 8 or hour > 18:
                location = home
            else:
                location = work
            
            # Add small GPS noise
            lat = location[0] + (random.random() - 0.5) * 0.00005
            lon = location[1] + (random.random() - 0.5) * 0.00005
            
            timestamp = base_time + timedelta(days=day, hours=hour)
            
            timestamps.append(timestamp)
            latitudes.append(lat)
            longitudes.append(lon)
    
    return {
        'timestamp': timestamps,
        'latitude': latitudes,
        'longitude': longitudes
    }


def example_basic_location_diversity():
    """Example showing basic location diversity extraction."""
    print("=== Basic Location Diversity Example ===")
    
    # Initialize with default configuration
    features = BehavioralActivationFeatures()
    
    # Create sample GPS data
    gps_data = create_realistic_gps_data(days=7, points_per_day=30)
    
    print(f"\nInput data summary:")
    print(f"  Total GPS points: {len(gps_data['timestamp'])}")
    print(f"  Time span: {(gps_data['timestamp'][-1] - gps_data['timestamp'][0]).days} days")
    print(f"  Date range: {gps_data['timestamp'][0].date()} to {gps_data['timestamp'][-1].date()}")
    
    # Extract location diversity
    result = features.location_diversity(gps_data)
    
    print(f"\nLocation Diversity Results:")
    diversity = result['location_diversity']
    print(f"  Weekly entropy: {diversity['weekly_entropy']:.3f} bits")
    print(f"  Cluster count: {diversity['cluster_count']}")
    print(f"  Unique locations: {diversity['unique_locations']}")
    print(f"  Max possible entropy: {diversity['max_possible_entropy']:.3f} bits")
    print(f"  Home cluster removed: {diversity['home_cluster_removed']}")
    
    # Show location probabilities
    print(f"\nLocation visitation patterns:")
    for cluster_id, prob_info in diversity['location_probabilities'].items():
        print(f"  Location {cluster_id}:")
        print(f"    Probability: {prob_info['probability']:.3f}")
        print(f"    Visit count: {prob_info['visit_count']}")
        print(f"    Center: ({prob_info['center_lat']:.4f}, {prob_info['center_lon']:.4f})")
    
    # Show quality metrics
    quality = result['quality_metrics']
    print(f"\nData Quality Metrics:")
    print(f"  Overall quality: {quality['overall_quality']:.3f}")
    print(f"  Coverage ratio: {quality['coverage_ratio']:.3f}")
    print(f"  Sampling rate: {quality['sampling_rate_per_day']:.1f} points/day")
    print(f"  Geographic area: {quality['geographic_area']:.2f} m²")
    print(f"  Coordinate consistency: {quality['coordinate_consistency']:.3f}")
    
    print()


def example_custom_configuration():
    """Example showing custom configuration for location diversity."""
    print("=== Custom Configuration Example ===")
    
    # Create custom configuration
    custom_config = LocationDiversityConfig(
        clustering_radius_meters=100.0,  # Larger clustering radius
        min_cluster_size=3,              # Smaller minimum cluster size
        analysis_window_days=14,         # 2-week analysis
        min_gps_points=50,               # Lower minimum points
        entropy_base=10.0,               # Base-10 entropy
        remove_home_location=False,      # Keep home location in analysis
        accuracy_weighting=False         # Don't weight by accuracy
    )
    
    features = BehavioralActivationFeatures(location_config=custom_config)
    
    print("Custom configuration parameters:")
    print(f"  Clustering radius: {custom_config.clustering_radius_meters} meters")
    print(f"  Minimum cluster size: {custom_config.min_cluster_size}")
    print(f"  Analysis window: {custom_config.analysis_window_days} days")
    print(f"  Minimum GPS points: {custom_config.min_gps_points}")
    print(f"  Entropy base: {custom_config.entropy_base}")
    print(f"  Remove home location: {custom_config.remove_home_location}")
    
    # Create GPS data
    gps_data = create_realistic_gps_data(days=14, points_per_day=20)
    
    # Extract with custom configuration
    result = features.location_diversity(gps_data)
    
    print(f"\nResults with custom configuration:")
    print(f"  Weekly entropy: {result['weekly_entropy']:.3f}")
    print(f"  Cluster count: {result['cluster_count']}")
    print(f"  Processing used {len(result['clusters'])} clusters")
    
    print()


def example_diversity_comparison():
    """Example comparing high vs low location diversity."""
    print("=== Location Diversity Comparison Example ===")
    
    # Initialize with lenient configuration for testing
    config = LocationDiversityConfig(
        min_gps_points=20,
        min_cluster_size=2,
        clustering_radius_meters=100.0
    )
    features = BehavioralActivationFeatures(location_config=config)
    
    # Test high diversity data
    print("Testing high diversity pattern...")
    high_diversity_data = create_high_diversity_data()
    high_result = features.location_diversity(high_diversity_data)
    
    print(f"  High diversity entropy: {high_result['weekly_entropy']:.3f}")
    print(f"  High diversity clusters: {high_result['cluster_count']}")
    
    # Test low diversity data
    print("\nTesting low diversity pattern...")
    low_diversity_data = create_low_diversity_data()
    low_result = features.location_diversity(low_diversity_data)
    
    print(f"  Low diversity entropy: {low_result['weekly_entropy']:.3f}")
    print(f"  Low diversity clusters: {low_result['cluster_count']}")
    
    # Comparison
    print(f"\nDiversity comparison:")
    entropy_diff = high_result['weekly_entropy'] - low_result['weekly_entropy']
    print(f"  Entropy difference: {entropy_diff:.3f} bits")
    print(f"  High diversity has {entropy_diff/low_result['weekly_entropy']*100:.1f}% more entropy")
    print(f"  Cluster difference: {high_result['cluster_count'] - low_result['cluster_count']}")
    
    print()


def example_clustering_analysis():
    """Example showing detailed clustering analysis."""
    print("=== Detailed Clustering Analysis Example ===")
    
    features = BehavioralActivationFeatures()
    
    # Create GPS data with clear patterns
    gps_data = create_realistic_gps_data(days=5, points_per_day=40)
    result = features.location_diversity(gps_data)
    
    print(f"Clustering Analysis Results:")
    clusters = result['clusters']
    
    print(f"  Total clusters found: {len(clusters)}")
    print(f"  Clusters after processing: {result['cluster_count']}")
    
    # Analyze each cluster
    print(f"\nDetailed cluster information:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {cluster['cluster_id']}:")
        print(f"    Center coordinates: ({cluster['center_latitude']:.4f}, {cluster['center_longitude']:.4f})")
        print(f"    Point count: {cluster['point_count']}")
        print(f"    Radius: {cluster['radius_meters']:.1f} meters")
        print(f"    Is home: {cluster['is_home']}")
        print(f"    Point indices: {cluster['point_indices'][:5]}{'...' if len(cluster['point_indices']) > 5 else ''}")
        
        # Estimate location type based on coordinates
        lat, lon = cluster['center_latitude'], cluster['center_longitude']
        if 40.71 <= lat <= 40.715 and -74.01 <= lon <= -74.00:
            location_type = "Likely Home (City Hall area)"
        elif 40.755 <= lat <= 40.761 and -73.99 <= lon <= -73.98:
            location_type = "Likely Work (Times Square area)"
        elif 40.745 <= lat <= 40.752 and -73.97 <= lon <= -73.965:
            location_type = "Likely Cafe/Transit (Grand Central area)"
        else:
            location_type = "Other location"
        
        print(f"    Estimated type: {location_type}")
    
    # Show clustering statistics
    total_points = sum(c['point_count'] for c in clusters)
    avg_cluster_size = total_points / len(clusters) if clusters else 0
    max_cluster_size = max(c['point_count'] for c in clusters) if clusters else 0
    min_cluster_size = min(c['point_count'] for c in clusters) if clusters else 0
    
    print(f"\nClustering statistics:")
    print(f"  Total points clustered: {total_points}")
    print(f"  Average cluster size: {avg_cluster_size:.1f}")
    print(f"  Largest cluster: {max_cluster_size} points")
    print(f"  Smallest cluster: {min_cluster_size} points")
    print(f"  Average cluster radius: {sum(c['radius_meters'] for c in clusters) / len(clusters):.1f} meters")
    
    print()


def example_quality_assessment():
    """Example showing GPS data quality assessment."""
    print("=== GPS Quality Assessment Example ===")
    
    features = BehavioralActivationFeatures()
    
    # Test different quality scenarios
    scenarios = [
        ("High Quality", create_realistic_gps_data(days=3, points_per_day=50)),
        ("Medium Quality", create_realistic_gps_data(days=3, points_per_day=20)),
        ("Low Quality", create_realistic_gps_data(days=3, points_per_day=5))
    ]
    
    for scenario_name, gps_data in scenarios:
        print(f"\n{scenario_name} GPS Data:")
        
        # Assess quality
        quality = features._assess_gps_quality(
            gps_data['timestamp'],
            gps_data['latitude'],
            gps_data['longitude']
        )
        
        print(f"  Points: {len(gps_data['timestamp'])}")
        print(f"  Overall quality: {quality['overall_quality']:.3f}")
        print(f"  Coverage ratio: {quality['coverage_ratio']:.3f}")
        print(f"  Sampling rate: {quality['sampling_rate_per_day']:.1f} points/day")
        print(f"  Geographic area: {quality['geographic_area']:.0f} m²")
        print(f"  Coordinate consistency: {quality['coordinate_consistency']:.3f}")
        
        # Try extraction with appropriate config
        min_points = min(10, len(gps_data['timestamp']))
        config = LocationDiversityConfig(
            min_gps_points=min_points,
            min_cluster_size=2,
            clustering_radius_meters=200.0
        )
        
        features_with_config = BehavioralActivationFeatures(location_config=config)
        
        try:
            result = features_with_config.location_diversity(gps_data)
            print(f"  Extraction successful: entropy = {result['weekly_entropy']:.3f}")
        except ValueError as e:
            print(f"  Extraction failed: {e}")
    
    print()


def example_entropy_interpretation():
    """Example showing how to interpret entropy values."""
    print("=== Entropy Interpretation Example ===")
    
    features = BehavioralActivationFeatures()
    
    # Create different patterns and calculate entropy
    patterns = [
        ("Single Location", [(40.7128, -74.0060)]),
        ("Two Locations", [(40.7128, -74.0060), (40.7580, -73.9855)]),
        ("Three Locations", [(40.7128, -74.0060), (40.7580, -73.9855), (40.7489, -73.9680)]),
        ("Four Locations", [(40.7128, -74.0060), (40.7580, -73.9855), (40.7489, -73.9680), (40.7829, -73.9654)])
    ]
    
    for pattern_name, locations in patterns:
        # Create GPS data for this pattern
        gps_data = {
            'timestamp': [],
            'latitude': [],
            'longitude': []
        }
        
        base_time = datetime(2026, 2, 21, 8, 0, 0)
        
        for day in range(7):
            for point in range(20):
                location_idx = point % len(locations)
                lat, lon = locations[location_idx]
                
                # Add small noise
                lat += (random.random() - 0.5) * 0.00005
                lon += (random.random() - 0.5) * 0.00005
                
                timestamp = base_time + timedelta(days=day, hours=8 + point * 12/20)
                
                gps_data['timestamp'].append(timestamp)
                gps_data['latitude'].append(lat)
                gps_data['longitude'].append(lon)
        
        # Extract location diversity
        config = LocationDiversityConfig(
            min_gps_points=10,
            min_cluster_size=2,
            clustering_radius_meters=100.0,
            remove_home_location=False
        )
        
        features_with_config = BehavioralActivationFeatures(location_config=config)
        result = features_with_config.location_diversity(gps_data)
        
        entropy = result['weekly_entropy']
        max_entropy = result['location_diversity']['max_possible_entropy']
        diversity_ratio = entropy / max_entropy if max_entropy > 0 else 0
        
        print(f"\n{pattern_name} Pattern:")
        print(f"  Entropy: {entropy:.3f} bits")
        print(f"  Max possible entropy: {max_entropy:.3f} bits")
        print(f"  Diversity ratio: {diversity_ratio:.3f}")
        
        # Interpretation
        if diversity_ratio < 0.3:
            interpretation = "Very low diversity - highly routine behavior"
        elif diversity_ratio < 0.6:
            interpretation = "Low to moderate diversity - somewhat routine"
        elif diversity_ratio < 0.8:
            interpretation = "Moderate to high diversity - balanced routine"
        else:
            interpretation = "High diversity - varied behavioral patterns"
        
        print(f"  Interpretation: {interpretation}")
    
    print()


def example_complete_analysis_workflow():
    """Example showing complete location diversity analysis workflow."""
    print("=== Complete Location Diversity Analysis Workflow ===")
    
    # Step 1: Configuration
    print("Step 1: Configuration Setup")
    config = LocationDiversityConfig(
        clustering_radius_meters=75.0,    # 75-meter clustering
        min_cluster_size=3,                # Minimum 3 points per cluster
        analysis_window_days=7,            # Weekly analysis
        min_gps_points=50,                 # Require 50 GPS points
        entropy_base=2.0,                  # Binary entropy (bits)
        remove_home_location=True,         # Exclude home from diversity
        accuracy_weighting=True           # Weight by GPS accuracy
    )
    
    features = BehavioralActivationFeatures(location_config=config)
    
    print(f"  Configuration: {config.clustering_radius_meters}m radius, {config.min_cluster_size} min points")
    
    # Step 2: Data Generation
    print("\nStep 2: Data Generation")
    gps_data = create_realistic_gps_data(days=7, points_per_day=40)
    
    print(f"  Generated {len(gps_data['timestamp'])} GPS points")
    print(f"  Time span: {(gps_data['timestamp'][-1] - gps_data['timestamp'][0]).days + 1} days")
    
    # Step 3: Feature Extraction
    print("\nStep 3: Location Diversity Extraction")
    result = features.location_diversity(gps_data)
    
    print(f"  ✓ Extraction completed successfully")
    print(f"  ✓ Found {result['cluster_count']} location clusters")
    print(f"  ✓ Calculated entropy: {result['weekly_entropy']:.3f} bits")
    
    # Step 4: Quality Assessment
    print("\nStep 4: Quality Assessment")
    quality = result['quality_metrics']
    
    print(f"  Data quality score: {quality['overall_quality']:.3f}/1.0")
    print(f"  Coverage ratio: {quality['coverage_ratio']:.1%}")
    print(f"  Sampling rate: {quality['sampling_rate_per_day']:.1f} points/day")
    
    if quality['overall_quality'] > 0.7:
        quality_assessment = "High quality data"
    elif quality['overall_quality'] > 0.4:
        quality_assessment = "Acceptable quality data"
    else:
        quality_assessment = "Low quality data - interpret with caution"
    
    print(f"  Assessment: {quality_assessment}")
    
    # Step 5: Results Interpretation
    print("\nStep 5: Results Interpretation")
    diversity = result['location_diversity']
    
    print(f"  Weekly entropy: {diversity['weekly_entropy']:.3f} bits")
    print(f"  Unique locations: {diversity['unique_locations']}")
    print(f"  Home cluster removed: {diversity['home_cluster_removed']}")
    
    # Behavioral interpretation
    entropy_per_location = diversity['weekly_entropy'] / diversity['unique_locations'] if diversity['unique_locations'] > 0 else 0
    
    if diversity['weekly_entropy'] < 1.0:
        behavioral_pattern = "Highly routine - strong location preferences"
        activation_level = "Lower behavioral activation"
    elif diversity['weekly_entropy'] < 2.0:
        behavioral_pattern = "Moderately routine - some location variety"
        activation_level = "Moderate behavioral activation"
    else:
        behavioral_pattern = "High variety - diverse location usage"
        activation_level = "Higher behavioral activation"
    
    print(f"  Behavioral pattern: {behavioral_pattern}")
    print(f"  Activation level: {activation_level}")
    print(f"  Entropy per location: {entropy_per_location:.3f} bits/location")
    
    # Step 6: Location Analysis
    print("\nStep 6: Location Pattern Analysis")
    clusters = result['clusters']
    
    # Sort clusters by visitation frequency
    sorted_clusters = sorted(clusters, key=lambda x: x['point_count'], reverse=True)
    
    print(f"  Top 3 most visited locations:")
    for i, cluster in enumerate(sorted_clusters[:3]):
        percentage = cluster['point_count'] / sum(c['point_count'] for c in clusters) * 100
        print(f"    {i+1}. {cluster['point_count']} visits ({percentage:.1f}%) - "
              f"Radius: {cluster['radius_meters']:.1f}m - Home: {cluster['is_home']}")
    
    # Step 7: Clinical Interpretation
    print("\nStep 7: Clinical Interpretation")
    
    if diversity['weekly_entropy'] > 2.5:
        clinical_note = "High location diversity suggests good environmental engagement and exploration"
    elif diversity['weekly_entropy'] > 1.5:
        clinical_note = "Moderate location diversity indicates balanced routine and exploration"
    elif diversity['weekly_entropy'] > 0.8:
        clinical_note = "Lower diversity may indicate structured routine or reduced environmental engagement"
    else:
        clinical_note = "Very low diversity may suggest significant behavioral restriction or withdrawal"
    
    print(f"  Clinical note: {clinical_note}")
    
    # Step 8: Data Summary
    print("\nStep 8: Processing Summary")
    processing = result['processing_parameters']
    data_summary = result['data_summary']
    
    print(f"  Processing parameters used:")
    for key, value in processing.items():
        print(f"    {key}: {value}")
    
    print(f"  Data summary:")
    for key, value in data_summary.items():
        print(f"    {key}: {value}")
    
    print()


if __name__ == "__main__":
    """Run all location diversity examples."""
    print("Psyconstruct Location Diversity Feature Examples")
    print("=" * 60)
    
    example_basic_location_diversity()
    example_custom_configuration()
    example_diversity_comparison()
    example_clustering_analysis()
    example_quality_assessment()
    example_entropy_interpretation()
    example_complete_analysis_workflow()
    
    print("All location diversity examples completed successfully!")
