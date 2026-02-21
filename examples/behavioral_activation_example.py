"""
Example usage of behavioral activation features.

This example demonstrates how to:
1. Extract activity volume from accelerometer data
2. Configure feature extraction parameters
3. Handle data quality issues
4. Interpret results and quality metrics
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import math
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psyconstruct.features.behavioral_activation import (
    BehavioralActivationFeatures,
    ActivityVolumeConfig
)


def create_realistic_accelerometer_data(days: int = 3, sampling_rate_hz: float = 1.0):
    """Create realistic accelerometer data for demonstration."""
    
    print(f"Generating {days} days of accelerometer data at {sampling_rate_hz} Hz...")
    
    timestamps = []
    x, y, z = [], [], []
    
    # Start from a specific date
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    # Generate data for each day
    for day in range(days):
        for hour in range(24):
            for second in range(int(3600 / sampling_rate_hz)):
                if second % int(1 / sampling_rate_hz) == 0:
                    ts = base_time + timedelta(days=day, hours=hour, seconds=second)
                    timestamps.append(ts)
                    
                    # Simulate realistic activity patterns
                    if 6 <= hour <= 8:  # Morning routine
                        activity = 0.8 + 0.3 * math.sin(hour * math.pi / 6)
                    elif 9 <= hour <= 17:  # Work hours
                        activity = 0.6 + 0.4 * math.sin((hour - 9) * math.pi / 8)
                    elif 18 <= hour <= 22:  # Evening activities
                        activity = 0.9 + 0.2 * math.sin((hour - 18) * math.pi / 4)
                    else:  # Sleep hours
                        activity = 0.1 + 0.05 * random.random()  # Small movements during sleep
                    
                    # Add some randomness
                    activity += (random.random() - 0.5) * 0.1
                    activity = max(0.1, activity)  # Ensure minimum activity
                    
                    # Generate accelerometer values
                    x_val = activity * 0.15 + (random.random() - 0.5) * 0.02
                    y_val = activity * 0.25 + (random.random() - 0.5) * 0.03
                    z_val = 9.8 + activity * 0.12 + (random.random() - 0.5) * 0.05
                    
                    x.append(x_val)
                    y.append(y_val)
                    z.append(z_val)
    
    return {
        'timestamp': timestamps,
        'x': x,
        'y': y,
        'z': z
    }


def create_problematic_data():
    """Create accelerometer data with quality issues for demonstration."""
    
    print("Generating problematic accelerometer data with gaps and outliers...")
    
    timestamps = []
    x, y, z = [], [], []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    # Generate data with gaps and issues
    for hour in range(24):
        if hour % 4 == 0:  # Only data every 4 hours (sparse)
            hour_start = base_time + timedelta(hours=hour)
            
            for minute in range(0, 60, 10):  # Every 10 minutes
                ts = hour_start + timedelta(minutes=minute)
                timestamps.append(ts)
                
                # Add some outliers
                if minute == 30:  # Add outlier at 30 minutes
                    x.append(10.0)  # Extreme outlier
                    y.append(10.0)
                    z.append(10.0)
                else:
                    x.append(0.1)
                    y.append(0.2)
                    z.append(9.8)
    
    return {
        'timestamp': timestamps,
        'x': x,
        'y': y,
        'z': z
    }


def example_basic_activity_volume():
    """Example showing basic activity volume extraction."""
    print("=== Basic Activity Volume Example ===")
    
    # Initialize features extractor
    features = BehavioralActivationFeatures()
    
    # Create sample data
    accel_data = create_realistic_accelerometer_data(days=2, sampling_rate_hz=0.5)
    
    print(f"Input data summary:")
    print(f"  Total records: {len(accel_data['timestamp'])}")
    print(f"  Time span: {(accel_data['timestamp'][-1] - accel_data['timestamp'][0]).days} days")
    print(f"  Sampling rate: {len(accel_data['timestamp']) / ((accel_data['timestamp'][-1] - accel_data['timestamp'][0]).total_seconds()):.2f} Hz")
    
    # Extract activity volume
    result = features.activity_volume(accel_data)
    
    print(f"\nActivity Volume Results:")
    activity_volumes = result['activity_volume']
    print(f"  Number of days analyzed: {len(activity_volumes)}")
    
    for i, av in enumerate(activity_volumes):
        print(f"  Day {i+1} ({av['date']}):")
        print(f"    Total volume: {av['volume']:.2f}")
        print(f"    Volume per hour: {av['volume_per_hour']:.2f}")
        print(f"    Sample count: {av['sample_count']}")
    
    # Show quality metrics
    quality = result['quality_metrics']
    print(f"\nData Quality Metrics:")
    print(f"  Coverage ratio: {quality['coverage_ratio']:.3f}")
    print(f"  Sampling rate: {quality['sampling_rate_hz']:.3f} Hz")
    print(f"  Max gap: {quality['gap_statistics']['max_gap_minutes']:.1f} minutes")
    print(f"  Outliers: {quality['outlier_statistics']['outlier_count']} ({quality['outlier_statistics']['outlier_percentage']:.1%})")
    print(f"  Overall quality: {quality['overall_quality']:.3f}")
    
    print()


def example_custom_configuration():
    """Example showing custom configuration for activity volume."""
    print("=== Custom Configuration Example ===")
    
    # Create custom configuration
    custom_config = ActivityVolumeConfig(
        window_hours=12,  # 12-hour windows instead of 24
        min_data_coverage=0.5,  # Lower coverage requirement
        min_sampling_rate_hz=0.05,  # Lower sampling rate requirement
        max_gap_minutes=120.0,  # Allow larger gaps
        outlier_threshold_std=2.5,  # More sensitive outlier detection
        interpolate_gaps=True,
        remove_outliers=True
    )
    
    features = BehavioralActivationFeatures(custom_config)
    
    print("Custom configuration parameters:")
    print(f"  Window hours: {custom_config.window_hours}")
    print(f"  Minimum coverage: {custom_config.min_data_coverage}")
    print(f"  Minimum sampling rate: {custom_config.min_sampling_rate_hz} Hz")
    print(f"  Maximum gap: {custom_config.max_gap_minutes} minutes")
    print(f"  Outlier threshold: {custom_config.outlier_threshold_std} std")
    
    # Create data with some quality issues
    accel_data = create_realistic_accelerometer_data(days=1, sampling_rate_hz=0.1)
    
    # Extract with custom configuration
    result = features.activity_volume(accel_data)
    
    print(f"\nResults with custom configuration:")
    print(f"  Processing parameters: {result['processing_parameters']}")
    print(f"  Activity volumes calculated: {len(result['activity_volume'])}")
    print(f"  Data quality: {result['quality_metrics']['overall_quality']:.3f}")
    
    print()


def example_quality_issues_handling():
    """Example showing handling of data quality issues."""
    print("=== Quality Issues Handling Example ===")
    
    # Initialize with strict configuration
    strict_config = ActivityVolumeConfig(
        min_data_coverage=0.8,
        min_sampling_rate_hz=0.5,
        remove_outliers=True,
        interpolate_gaps=True
    )
    
    strict_features = BehavioralActivationFeatures(strict_config)
    
    # Initialize with lenient configuration
    lenient_config = ActivityVolumeConfig(
        min_data_coverage=0.3,
        min_sampling_rate_hz=0.05,
        remove_outliers=False,
        interpolate_gaps=True
    )
    
    lenient_features = BehavioralActivationFeatures(lenient_config)
    
    # Create problematic data
    problematic_data = create_problematic_data()
    
    print(f"Problematic data summary:")
    print(f"  Total records: {len(problematic_data['timestamp'])}")
    print(f"  Time span: {(problematic_data['timestamp'][-1] - problematic_data['timestamp'][0]).total_seconds() / 3600:.1f} hours")
    
    # Try with strict configuration (should fail)
    print("\nTrying strict configuration...")
    try:
        strict_result = strict_features.activity_volume(problematic_data)
        print("  Success! (Unexpected)")
    except ValueError as e:
        print(f"  Failed as expected: {e}")
    
    # Try with lenient configuration (should succeed)
    print("\nTrying lenient configuration...")
    try:
        lenient_result = lenient_features.activity_volume(problematic_data)
        print("  Success!")
        
        quality = lenient_result['quality_metrics']
        print(f"  Coverage ratio: {quality['coverage_ratio']:.3f}")
        print(f"  Outliers detected: {quality['outlier_statistics']['outlier_count']}")
        print(f"  Overall quality: {quality['overall_quality']:.3f}")
        
    except ValueError as e:
        print(f"  Failed: {e}")
    
    print()


def example_time_window_analysis():
    """Example showing analysis with specific time windows."""
    print("=== Time Window Analysis Example ===")
    
    features = BehavioralActivationFeatures()
    
    # Create 5 days of data
    accel_data = create_realistic_accelerometer_data(days=5, sampling_rate_hz=0.25)
    
    # Define different analysis windows
    windows = [
        {
            'name': 'First 24 hours',
            'start': datetime(2026, 2, 21, 0, 0, 0),
            'end': datetime(2026, 2, 22, 0, 0, 0)
        },
        {
            'name': 'Middle 48 hours',
            'start': datetime(2026, 2, 22, 0, 0, 0),
            'end': datetime(2026, 2, 24, 0, 0, 0)
        },
        {
            'name': 'Last 24 hours',
            'start': datetime(2026, 2, 24, 0, 0, 0),
            'end': datetime(2026, 2, 25, 0, 0, 0)
        }
    ]
    
    for window in windows:
        print(f"\nAnalyzing {window['name']}:")
        print(f"  Window: {window['start']} to {window['end']}")
        
        result = features.activity_volume(
            accel_data,
            window_start=window['start'],
            window_end=window['end']
        )
        
        activity_volumes = result['activity_volume']
        summary = result['data_summary']
        
        print(f"  Days analyzed: {len(activity_volumes)}")
        print(f"  Total records in window: {summary['total_records']}")
        
        if activity_volumes:
            volumes = [av['volume'] for av in activity_volumes]
            print(f"  Average daily volume: {sum(volumes) / len(volumes):.2f}")
            print(f"  Volume range: {min(volumes):.2f} - {max(volumes):.2f}")
    
    print()


def example_detailed_quality_analysis():
    """Example showing detailed quality analysis."""
    print("=== Detailed Quality Analysis Example ===")
    
    features = BehavioralActivationFeatures()
    
    # Create data with varying quality
    print("Creating datasets with different quality characteristics...")
    
    # High quality dataset
    high_quality_data = create_realistic_accelerometer_data(days=1, sampling_rate_hz=1.0)
    
    # Medium quality dataset
    medium_quality_data = create_realistic_accelerometer_data(days=1, sampling_rate_hz=0.2)
    
    # Low quality dataset (with gaps)
    low_quality_data = create_realistic_accelerometer_data(days=1, sampling_rate_hz=0.05)
    
    datasets = [
        ('High Quality', high_quality_data),
        ('Medium Quality', medium_quality_data),
        ('Low Quality', low_quality_data)
    ]
    
    for name, data in datasets:
        print(f"\n{name} Dataset Analysis:")
        
        try:
            result = features.activity_volume(data)
            quality = result['quality_metrics']
            
            print(f"  ✓ Extraction successful")
            print(f"  Coverage ratio: {quality['coverage_ratio']:.3f}")
            print(f"  Sampling rate: {quality['sampling_rate_hz']:.3f} Hz")
            print(f"  Max gap: {quality['gap_statistics']['max_gap_minutes']:.1f} min")
            print(f"  Outlier count: {quality['outlier_statistics']['outlier_count']}")
            print(f"  Overall quality: {quality['overall_quality']:.3f}")
            
            if result['activity_volume']:
                av = result['activity_volume'][0]
                print(f"  Activity volume: {av['volume']:.2f}")
                print(f"  Samples used: {av['sample_count']}")
        
        except ValueError as e:
            print(f"  ✗ Extraction failed: {e}")
    
    print()


def example_activity_patterns():
    """Example showing different activity patterns."""
    print("=== Activity Patterns Example ===")
    
    features = BehavioralActivationFeatures()
    
    # Simulate different activity patterns
    patterns = [
        {
            'name': 'High Activity Day',
            'description': 'Person with high behavioral activation',
            'activity_multiplier': 2.0
        },
        {
            'name': 'Low Activity Day', 
            'description': 'Person with low behavioral activation',
            'activity_multiplier': 0.3
        },
        {
            'name': 'Normal Activity Day',
            'description': 'Person with normal activity patterns',
            'activity_multiplier': 1.0
        }
    ]
    
    for pattern in patterns:
        print(f"\n{pattern['name']}:")
        print(f"  {pattern['description']}")
        
        # Create data with specific activity level
        base_data = create_realistic_accelerometer_data(days=1, sampling_rate_hz=0.5)
        
        # Adjust activity levels
        multiplier = pattern['activity_multiplier']
        adjusted_data = {
            'timestamp': base_data['timestamp'],
            'x': [x * multiplier for x in base_data['x']],
            'y': [y * multiplier for y in base_data['y']],
            'z': [9.8 + (z - 9.8) * multiplier for z in base_data['z']]
        }
        
        # Extract activity volume
        result = features.activity_volume(adjusted_data)
        
        if result['activity_volume']:
            av = result['activity_volume'][0]
            print(f"  Activity volume: {av['volume']:.2f}")
            print(f"  Volume per hour: {av['volume_per_hour']:.2f}")
            print(f"  Sample count: {av['sample_count']}")
            print(f"  Data quality: {result['quality_metrics']['overall_quality']:.3f}")
    
    print()


def example_complete_analysis_workflow():
    """Example showing complete analysis workflow."""
    print("=== Complete Analysis Workflow Example ===")
    
    # Initialize with comprehensive configuration
    config = ActivityVolumeConfig(
        window_hours=24,
        min_data_coverage=0.6,
        min_sampling_rate_hz=0.1,
        max_gap_minutes=120.0,
        outlier_threshold_std=3.0,
        interpolate_gaps=True,
        remove_outliers=True
    )
    
    features = BehavioralActivationFeatures(config)
    
    print("Step 1: Data Generation")
    print("  Creating 7 days of accelerometer data...")
    accel_data = create_realistic_accelerometer_data(days=7, sampling_rate_hz=0.25)
    
    print(f"  Generated {len(accel_data['timestamp'])} records")
    print(f"  Time span: {(accel_data['timestamp'][-1] - accel_data['timestamp'][0]).days} days")
    
    print("\nStep 2: Feature Extraction")
    print("  Extracting activity volume...")
    
    # Extract activity volume for entire dataset
    result = features.activity_volume(accel_data)
    
    print(f"  ✓ Successfully extracted activity volumes")
    print(f"  ✓ Analyzed {len(result['activity_volume'])} days")
    
    print("\nStep 3: Quality Assessment")
    quality = result['quality_metrics']
    print(f"  Data coverage: {quality['coverage_ratio']:.1%}")
    print(f"  Sampling quality: {quality['sampling_rate_hz']:.2f} Hz")
    print(f"  Gap analysis: {quality['gap_statistics']['max_gap_minutes']:.1f} min max gap")
    print(f"  Outlier analysis: {quality['outlier_statistics']['outlier_count']} outliers detected")
    print(f"  Overall quality score: {quality['overall_quality']:.3f}/1.0")
    
    print("\nStep 4: Results Summary")
    activity_volumes = result['activity_volume']
    volumes = [av['volume'] for av in activity_volumes]
    
    print(f"  Analysis period: {activity_volumes[0]['date']} to {activity_volumes[-1]['date']}")
    print(f"  Total activity volume: {sum(volumes):.2f}")
    print(f"  Daily average: {sum(volumes) / len(volumes):.2f}")
    print(f"  Daily range: {min(volumes):.2f} - {max(volumes):.2f}")
    print(f"  Standard deviation: {math.sqrt(sum((v - sum(volumes)/len(volumes))**2 for v in volumes) / len(volumes)):.2f}")
    
    print("\nStep 5: Day-by-Day Breakdown")
    for i, av in enumerate(activity_volumes):
        weekday = datetime.fromisoformat(av['date']).strftime('%A')
        print(f"  Day {i+1} ({av['date']}, {weekday}):")
        print(f"    Volume: {av['volume']:.2f} ({av['volume_per_hour']:.2f}/hour)")
        print(f"    Samples: {av['sample_count']}")
        print(f"    Quality: {'High' if av['volume'] > sum(volumes)/len(volumes) else 'Low'} activity")
    
    print("\nStep 6: Interpretation")
    avg_volume = sum(volumes) / len(volumes)
    high_activity_days = sum(1 for v in volumes if v > avg_volume * 1.2)
    low_activity_days = sum(1 for v in volumes if v < avg_volume * 0.8)
    
    print(f"  Behavioral activation assessment:")
    print(f"    Average daily activity: {avg_volume:.2f}")
    print(f"    High activity days: {high_activity_days}/{len(volumes)}")
    print(f"    Low activity days: {low_activity_days}/{len(volumes)}")
    print(f"    Activity consistency: {'High' if high_activity_days + low_activity_days < 3 else 'Variable'}")
    
    print(f"\nStep 7: Data Processing Summary")
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
    """Run all behavioral activation examples."""
    print("Psyconstruct Behavioral Activation Features Examples")
    print("=" * 65)
    
    example_basic_activity_volume()
    example_custom_configuration()
    example_quality_issues_handling()
    example_time_window_analysis()
    example_detailed_quality_analysis()
    example_activity_patterns()
    example_complete_analysis_workflow()
    
    print("All behavioral activation examples completed successfully!")
