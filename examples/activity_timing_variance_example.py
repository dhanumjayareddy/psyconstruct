"""
Example usage of activity timing variance feature.

This example demonstrates how to:
1. Extract activity timing variance from accelerometer data
2. Configure variance calculation and timing parameters
3. Interpret timing variance results
4. Handle different activity pattern scenarios
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
    ActivityTimingVarianceConfig
)


def create_regular_activity_data(days: int = 7, samples_per_hour: int = 120):
    """Create accelerometer data with regular activity patterns."""
    
    print(f"Generating {days} days of regular activity data with {samples_per_hour} samples/hour...")
    
    timestamps = []
    x_values = []
    y_values = []
    z_values = []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    for day in range(days):
        for hour in range(24):
            for sample in range(samples_per_hour):
                minute = sample * 60 // samples_per_hour
                second = (sample * 60 % samples_per_hour) * 60 // samples_per_hour
                
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute, seconds=second)
                
                # Regular activity pattern
                if 6 <= hour <= 8:  # Morning routine
                    activity_level = 2.0 + random.random() * 0.5
                elif 9 <= hour <= 12:  # Morning work
                    activity_level = 1.5 + random.random() * 0.3
                elif 13 <= hour <= 14:  # Lunch break
                    activity_level = 2.5 + random.random() * 0.5
                elif 15 <= hour <= 17:  # Afternoon work
                    activity_level = 1.3 + random.random() * 0.3
                elif 18 <= hour <= 20:  # Evening activity
                    activity_level = 2.2 + random.random() * 0.4
                elif 21 <= hour <= 22:  # Wind down
                    activity_level = 1.0 + random.random() * 0.3
                else:  # Sleep
                    activity_level = 0.1 + random.random() * 0.1
                
                # Generate accelerometer values
                x = activity_level * (random.random() - 0.5) * 2
                y = activity_level * (random.random() - 0.5) * 2
                z = 9.8 + activity_level * random.random()
                
                timestamps.append(timestamp)
                x_values.append(x)
                y_values.append(y)
                z_values.append(z)
    
    return {
        'timestamp': timestamps,
        'x': x_values,
        'y': y_values,
        'z': z_values
    }


def create_irregular_activity_data(days: int = 7, samples_per_hour: int = 120):
    """Create accelerometer data with irregular activity patterns."""
    
    print(f"Generating {days} days of irregular activity data with {samples_per_hour} samples/hour...")
    
    timestamps = []
    x_values = []
    y_values = []
    z_values = []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    for day in range(days):
        # Randomize the daily pattern
        day_type = random.choice(['early_bird', 'night_owl', 'irregular', 'mixed'])
        
        for hour in range(24):
            for sample in range(samples_per_hour):
                minute = sample * 60 // samples_per_hour
                second = (sample * 60 % samples_per_hour) * 60 // samples_per_hour
                
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute, seconds=second)
                
                # Irregular activity pattern based on day type
                if day_type == 'early_bird':
                    if 5 <= hour <= 7:  # Very early morning
                        activity_level = 2.5 + random.random() * 0.5
                    elif 8 <= hour <= 16:  # Daytime
                        activity_level = 1.8 + random.random() * 0.4
                    elif 20 <= hour <= 22:  # Evening
                        activity_level = 0.8 + random.random() * 0.3
                    else:
                        activity_level = 0.1 + random.random() * 0.1
                        
                elif day_type == 'night_owl':
                    if 0 <= hour <= 2:  # Late night
                        activity_level = 2.0 + random.random() * 0.5
                    elif 10 <= hour <= 12:  # Late morning
                        activity_level = 1.5 + random.random() * 0.3
                    elif 14 <= hour <= 18:  # Afternoon
                        activity_level = 2.2 + random.random() * 0.4
                    elif 22 <= hour <= 23:  # Late evening
                        activity_level = 2.5 + random.random() * 0.5
                    else:
                        activity_level = 0.1 + random.random() * 0.1
                        
                elif day_type == 'irregular':
                    # Completely random activity
                    if random.random() < 0.3:  # 30% chance of high activity
                        activity_level = 2.0 + random.random() * 1.0
                    else:
                        activity_level = 0.1 + random.random() * 0.5
                        
                else:  # mixed
                    # Mix of different patterns
                    if hour % 6 == 0:  # Every 6 hours
                        activity_level = 2.5 + random.random() * 0.5
                    elif random.random() < 0.2:
                        activity_level = 1.5 + random.random() * 0.5
                    else:
                        activity_level = 0.3 + random.random() * 0.3
                
                # Generate accelerometer values
                x = activity_level * (random.random() - 0.5) * 2
                y = activity_level * (random.random() - 0.5) * 2
                z = 9.8 + activity_level * random.random()
                
                timestamps.append(timestamp)
                x_values.append(x)
                y_values.append(y)
                z_values.append(z)
    
    return {
        'timestamp': timestamps,
        'x': x_values,
        'y': y_values,
        'z': z_values
    }


def create_shift_worker_data(days: int = 14, samples_per_hour: int = 120):
    """Create accelerometer data for shift worker with rotating schedules."""
    
    print(f"Generating {days} days of shift worker data with {samples_per_hour} samples/hour...")
    
    timestamps = []
    x_values = []
    y_values = []
    z_values = []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    # Shift patterns: day shift, evening shift, night shift
    shift_patterns = [
        {'name': 'day_shift', 'active_hours': [(6, 8), (9, 12), (13, 17), (18, 20)]},
        {'name': 'evening_shift', 'active_hours': [(10, 12), (14, 18), (19, 23), (0, 2)]},
        {'name': 'night_shift', 'active_hours': [(14, 16), (18, 22), (23, 3), (4, 7)]}
    ]
    
    for day in range(days):
        # Rotate through shift patterns
        shift = shift_patterns[day % len(shift_patterns)]
        
        for hour in range(24):
            for sample in range(samples_per_hour):
                minute = sample * 60 // samples_per_hour
                second = (sample * 60 % samples_per_hour) * 60 // samples_per_hour
                
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute, seconds=second)
                
                # Check if current hour is in active hours for this shift
                is_active = False
                for start, end in shift['active_hours']:
                    if start <= hour < end or (start > end and (hour >= start or hour < end)):
                        is_active = True
                        break
                
                if is_active:
                    activity_level = 2.0 + random.random() * 0.8
                else:
                    activity_level = 0.1 + random.random() * 0.2
                
                # Generate accelerometer values
                x = activity_level * (random.random() - 0.5) * 2
                y = activity_level * (random.random() - 0.5) * 2
                z = 9.8 + activity_level * random.random()
                
                timestamps.append(timestamp)
                x_values.append(x)
                y_values.append(y)
                z_values.append(z)
    
    return {
        'timestamp': timestamps,
        'x': x_values,
        'y': y_values,
        'z': z_values
    }


def example_basic_activity_timing_variance():
    """Example showing basic activity timing variance extraction."""
    print("=== Basic Activity Timing Variance Example ===")
    
    # Initialize with default configuration
    features = BehavioralActivationFeatures()
    
    # Create sample accelerometer data
    accel_data = create_regular_activity_data(days=7, samples_per_hour=60)
    
    print(f"\nInput data summary:")
    print(f"  Total accelerometer readings: {len(accel_data['timestamp'])}")
    print(f"  Time span: {(accel_data['timestamp'][-1] - accel_data['timestamp'][0]).days + 1} days")
    print(f"  Date range: {accel_data['timestamp'][0].date()} to {accel_data['timestamp'][-1].date()}")
    print(f"  Sampling rate: {len(accel_data['timestamp']) / ((accel_data['timestamp'][-1] - accel_data['timestamp'][0]).total_seconds() / 3600):.1f} samples/hour")
    
    # Extract activity timing variance
    result = features.activity_timing_variance(accel_data)
    
    print(f"\nActivity Timing Variance Results:")
    variance = result['activity_timing_variance']
    print(f"  Weekly variance: {variance['weekly_variance']:.4f}")
    print(f"  Peak activity hour: {variance['peak_activity_hour']}:00")
    print(f"  Variance stability: {variance['variance_stability']:.3f}")
    print(f"  Mean hourly variance: {variance['mean_hourly_variance']:.4f}")
    print(f"  Max hourly variance: {variance['max_hourly_variance']:.4f}")
    print(f"  Min hourly variance: {variance['min_hourly_variance']:.4f}")
    
    # Show hourly patterns
    print(f"\nHourly Variance Patterns (first 12 hours):")
    hourly_variances = variance['hourly_patterns']['hourly_variances']
    for hour in range(12):
        print(f"  {hour:02d}:00 - {hourly_variances[hour]:.4f}")
    
    # Show quality metrics
    quality = result['quality_metrics']
    print(f"\nData Quality Metrics:")
    print(f"  Overall quality: {quality['overall_quality']:.3f}")
    print(f"  Coverage ratio: {quality['coverage_ratio']:.3f}")
    print(f"  Sampling rate: {quality['sampling_rate_hz']:.1f} Hz")
    print(f"  Data completeness: {quality['data_completeness']:.3f}")
    print(f"  Temporal consistency: {quality['temporal_consistency']:.3f}")
    
    print()


def example_custom_configuration():
    """Example showing custom configuration for activity timing variance."""
    print("=== Custom Configuration Example ===")
    
    # Create custom configuration
    custom_config = ActivityTimingVarianceConfig(
        analysis_window_days=14,           # 2-week analysis
        time_resolution_minutes=30,        # 30-minute bins
        min_activity_threshold=0.2,        # Higher activity threshold
        min_active_hours_per_day=6,        # Require 6 active hours
        variance_metric="cv",              # Coefficient of variation
        include_weekend_analysis=True,     # Separate weekday/weekend
        normalize_by_activity_level=False, # Don't normalize
        smooth_activity_data=False,        # No smoothing
        min_days_with_data=7,              # Require 7 days
        min_data_coverage=0.8              # 80% coverage required
    )
    
    features = BehavioralActivationFeatures(timing_config=custom_config)
    
    print("Custom configuration parameters:")
    print(f"  Analysis window: {custom_config.analysis_window_days} days")
    print(f"  Time resolution: {custom_config.time_resolution_minutes} minutes")
    print(f"  Variance metric: {custom_config.variance_metric}")
    print(f"  Activity threshold: {custom_config.min_activity_threshold}")
    print(f"  Min active hours: {custom_config.min_active_hours_per_day}")
    print(f"  Smoothing: {custom_config.smooth_activity_data}")
    print(f"  Normalization: {custom_config.normalize_by_activity_level}")
    
    # Create accelerometer data
    accel_data = create_regular_activity_data(days=14, samples_per_hour=60)
    
    # Extract with custom configuration
    result = features.activity_timing_variance(accel_data)
    
    print(f"\nResults with custom configuration:")
    print(f"  Weekly variance: {result['weekly_variance']:.4f}")
    print(f"  Peak activity hour: {result['activity_timing_variance']['peak_activity_hour']}:00")
    print(f"  Variance stability: {result['activity_timing_variance']['variance_stability']:.3f}")
    print(f"  Time bins: {len(result['hourly_patterns']['hourly_variances'])}")
    
    # Show weekday/weekend analysis
    if 'variance_by_day_type' in result and result['variance_by_day_type']:
        variance_by_type = result['variance_by_day_type']
        print(f"\nWeekday/Weekend Analysis:")
        print(f"  Weekday variance: {variance_by_type.get('weekday', 'N/A'):.4f}")
        print(f"  Weekend variance: {variance_by_type.get('weekend', 'N/A'):.4f}")
        print(f"  Difference: {variance_by_type.get('weekday_weekend_difference', 'N/A'):.4f}")
    
    print()


def example_pattern_comparison():
    """Example comparing different activity patterns."""
    print("=== Activity Pattern Comparison Example ===")
    
    # Initialize with lenient configuration for testing
    config = ActivityTimingVarianceConfig(
        min_days_with_data=5,
        min_active_hours_per_day=3,
        time_resolution_minutes=60
    )
    features = BehavioralActivationFeatures(timing_config=config)
    
    # Test regular pattern
    print("Testing regular activity pattern...")
    regular_data = create_regular_activity_data(days=7, samples_per_hour=60)
    regular_result = features.activity_timing_variance(regular_data)
    
    print(f"  Regular pattern variance: {regular_result['weekly_variance']:.4f}")
    print(f"  Regular pattern stability: {regular_result['activity_timing_variance']['variance_stability']:.3f}")
    
    # Test irregular pattern
    print("\nTesting irregular activity pattern...")
    irregular_data = create_irregular_activity_data(days=7, samples_per_hour=60)
    irregular_result = features.activity_timing_variance(irregular_data)
    
    print(f"  Irregular pattern variance: {irregular_result['weekly_variance']:.4f}")
    print(f"  Irregular pattern stability: {irregular_result['activity_timing_variance']['variance_stability']:.3f}")
    
    # Comparison
    print(f"\nPattern comparison:")
    variance_diff = irregular_result['weekly_variance'] - regular_result['weekly_variance']
    stability_diff = regular_result['activity_timing_variance']['variance_stability'] - irregular_result['activity_timing_variance']['variance_stability']
    
    print(f"  Variance difference: {variance_diff:.4f}")
    print(f"  Stability difference: {stability_diff:.3f}")
    
    if variance_diff > 0:
        print(f"  Irregular pattern has {variance_diff/regular_result['weekly_variance']*100:.1f}% more variance")
    if stability_diff > 0:
        print(f"  Regular pattern is {stability_diff*100:.1f}% more stable")
    
    print()


def example_variance_metrics_comparison():
    """Example comparing different variance calculation methods."""
    print("=== Variance Metrics Comparison Example ===")
    
    # Create test data
    accel_data = create_regular_activity_data(days=7, samples_per_hour=60)
    
    # Test different variance metrics
    metrics = ['std', 'cv', 'iqr']
    results = {}
    
    for metric in metrics:
        print(f"\nTesting {metric.upper()} variance metric:")
        
        config = ActivityTimingVarianceConfig(
            variance_metric=metric,
            min_days_with_data=5,
            min_active_hours_per_day=3
        )
        
        features = BehavioralActivationFeatures(timing_config=config)
        result = features.activity_timing_variance(accel_data)
        
        results[metric] = result
        
        print(f"  Weekly variance: {result['weekly_variance']:.4f}")
        print(f"  Peak activity hour: {result['activity_timing_variance']['peak_activity_hour']}:00")
        print(f"  Variance stability: {result['activity_timing_variance']['variance_stability']:.3f}")
    
    # Comparison
    print(f"\nVariance Metrics Comparison:")
    for metric, result in results.items():
        print(f"  {metric.upper()}: {result['weekly_variance']:.4f}")
    
    # Show relative differences
    std_result = results['std']['weekly_variance']
    print(f"\nRelative to Standard Deviation:")
    for metric, result in results.items():
        if metric != 'std':
            ratio = result['weekly_variance'] / std_result if std_result > 0 else 0
            print(f"  {metric.upper()}: {ratio:.2f}x")
    
    print()


def example_shift_worker_analysis():
    """Example analyzing shift worker timing patterns."""
    print("=== Shift Worker Analysis Example ===")
    
    config = ActivityTimingVarianceConfig(
        analysis_window_days=14,
        include_weekend_analysis=True,
        time_resolution_minutes=60,
        min_days_with_data=7,
        min_active_hours_per_day=4
    )
    
    features = BehavioralActivationFeatures(timing_config=config)
    
    # Create shift worker data
    shift_data = create_shift_worker_data(days=14, samples_per_hour=60)
    
    print(f"Shift worker data generated:")
    print(f"  Total days: 14")
    print(f"  Shift pattern: Day → Evening → Night rotation")
    
    # Extract timing variance
    result = features.activity_timing_variance(shift_data)
    
    print(f"\nShift Worker Timing Analysis:")
    variance = result['activity_timing_variance']
    print(f"  Weekly variance: {variance['weekly_variance']:.4f}")
    print(f"  Peak activity hour: {variance['peak_activity_hour']}:00")
    print(f"  Variance stability: {variance['variance_stability']:.3f}")
    
    # High variance indicates irregular patterns
    if variance['weekly_variance'] > 0.05:
        print(f"  Pattern: High variance indicates irregular shift work")
    elif variance['weekly_variance'] > 0.02:
        print(f"  Pattern: Moderate variance indicates some schedule variation")
    else:
        print(f"  Pattern: Low variance indicates consistent schedule")
    
    # Show weekday/weekend differences
    if 'variance_by_day_type' in result and result['variance_by_day_type']:
        variance_by_type = result['variance_by_day_type']
        print(f"\nDay Type Analysis:")
        print(f"  Weekday variance: {variance_by_type.get('weekday', 'N/A'):.4f}")
        print(f"  Weekend variance: {variance_by_type.get('weekend', 'N/A'):.4f}")
        print(f"  Difference: {variance_by_type.get('weekday_weekend_difference', 'N/A'):.4f}")
    
    # Show hourly variance pattern
    print(f"\nHourly Variance Pattern (key hours):")
    hourly_variances = variance['hourly_patterns']['hourly_variances']
    key_hours = [6, 9, 12, 15, 18, 21, 0, 3]
    
    for hour in key_hours:
        if hour < len(hourly_variances):
            print(f"  {hour:02d}:00 - {hourly_variances[hour]:.4f}")
    
    print()


def example_time_resolution_analysis():
    """Example showing different time resolutions."""
    print("=== Time Resolution Analysis Example ===")
    
    # Create test data
    accel_data = create_regular_activity_data(days=7, samples_per_hour=120)
    
    # Test different time resolutions
    resolutions = [30, 60, 120]  # 30min, 1hour, 2hour
    
    for resolution in resolutions:
        print(f"\nTesting {resolution}-minute resolution:")
        
        config = ActivityTimingVarianceConfig(
            time_resolution_minutes=resolution,
            min_days_with_data=5,
            min_active_hours_per_day=2
        )
        
        features = BehavioralActivationFeatures(timing_config=config)
        result = features.activity_timing_variance(accel_data)
        
        print(f"  Time bins: {len(result['hourly_patterns']['hourly_variances'])}")
        print(f"  Weekly variance: {result['weekly_variance']:.4f}")
        print(f"  Peak activity bin: {result['activity_timing_variance']['peak_activity_hour']}")
        
        # Convert bin to actual time
        peak_hour = result['activity_timing_variance']['peak_activity_hour']
        actual_hour = (peak_hour * resolution) // 60
        actual_minute = (peak_hour * resolution) % 60
        print(f"  Peak activity time: {actual_hour:02d}:{actual_minute:02d}")
    
    print()


def example_quality_assessment():
    """Example showing timing data quality assessment."""
    print("=== Timing Data Quality Assessment Example ===")
    
    features = BehavioralActivationFeatures()
    
    # Test different quality scenarios
    scenarios = [
        ("High Quality", create_regular_activity_data(days=5, samples_per_hour=120)),
        ("Medium Quality", create_regular_activity_data(days=5, samples_per_hour=60)),
        ("Low Quality", create_regular_activity_data(days=5, samples_per_hour=20))
    ]
    
    for scenario_name, accel_data in scenarios:
        print(f"\n{scenario_name} Accelerometer Data:")
        
        # Calculate magnitude for quality assessment
        magnitude = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(
            accel_data['x'], accel_data['y'], accel_data['z']
        )]
        
        # Assess quality
        quality = features._assess_timing_data_quality(
            accel_data['timestamp'], magnitude
        )
        
        print(f"  Total readings: {len(accel_data['timestamp'])}")
        print(f"  Overall quality: {quality['overall_quality']:.3f}")
        print(f"  Coverage ratio: {quality['coverage_ratio']:.3f}")
        print(f"  Sampling rate: {quality['sampling_rate_hz']:.1f} Hz")
        print(f"  Data completeness: {quality['data_completeness']:.3f}")
        print(f"  Temporal consistency: {quality['temporal_consistency']:.3f}")
        
        # Try extraction with appropriate config
        min_active_hours = max(2, len(accel_data['timestamp']) // 1000)
        config = ActivityTimingVarianceConfig(
            min_days_with_data=3,
            min_active_hours_per_day=min_active_hours,
            time_resolution_minutes=60
        )
        
        features_with_config = BehavioralActivationFeatures(timing_config=config)
        
        try:
            result = features_with_config.activity_timing_variance(accel_data)
            print(f"  Extraction successful: variance = {result['weekly_variance']:.4f}")
        except ValueError as e:
            print(f"  Extraction failed: {e}")
    
    print()


def example_variance_interpretation():
    """Example showing how to interpret variance values."""
    print("=== Variance Interpretation Example ===")
    
    features = BehavioralActivationFeatures()
    
    # Create different patterns and calculate variance
    patterns = [
        ("Very Regular", create_regular_activity_data(days=7, samples_per_hour=60)),
        ("Moderately Regular", create_regular_activity_data(days=7, samples_per_hour=40)),
        ("Irregular", create_irregular_activity_data(days=7, samples_per_hour=60)),
        ("Very Irregular", create_shift_worker_data(days=7, samples_per_hour=60))
    ]
    
    config = ActivityTimingVarianceConfig(
        min_days_with_data=5,
        min_active_hours_per_day=2,
        time_resolution_minutes=60
    )
    
    features_with_config = BehavioralActivationFeatures(timing_config=config)
    
    for pattern_name, accel_data in patterns:
        result = features_with_config.activity_timing_variance(accel_data)
        variance = result['weekly_variance']
        stability = result['activity_timing_variance']['variance_stability']
        
        print(f"\n{pattern_name} Pattern:")
        print(f"  Variance: {variance:.4f}")
        print(f"  Stability: {stability:.3f}")
        
        # Interpretation
        if variance < 0.01:
            variance_interpretation = "Very consistent - highly regular routine"
        elif variance < 0.03:
            variance_interpretation = "Consistent - regular daily patterns"
        elif variance < 0.06:
            variance_interpretation = "Moderately variable - some schedule flexibility"
        elif variance < 0.10:
            variance_interpretation = "Variable - irregular patterns or schedule changes"
        else:
            variance_interpretation = "Highly variable - very irregular or chaotic patterns"
        
        print(f"  Interpretation: {variance_interpretation}")
        
        # Behavioral implications
        if stability > 0.8:
            behavioral_note = "Stable patterns suggest good routine adherence"
        elif stability > 0.5:
            behavioral_note = "Moderate stability indicates some routine flexibility"
        else:
            behavioral_note = "Low stability suggests irregular behavioral patterns"
        
        print(f"  Behavioral note: {behavioral_note}")
    
    print()


def example_complete_analysis_workflow():
    """Example showing complete activity timing variance analysis workflow."""
    print("=== Complete Activity Timing Variance Analysis Workflow ===")
    
    # Step 1: Configuration
    print("Step 1: Configuration Setup")
    config = ActivityTimingVarianceConfig(
        analysis_window_days=7,            # Weekly analysis
        time_resolution_minutes=60,        # 1-hour bins
        min_activity_threshold=0.1,        # Low activity threshold
        min_active_hours_per_day=4,        # 4 active hours minimum
        variance_metric="std",              # Standard deviation
        include_weekend_analysis=True,     # Include weekday/weekend
        normalize_by_activity_level=True,  # Normalize by activity
        smooth_activity_data=True,         # Apply smoothing
        min_days_with_data=5,              # 5 days minimum
        min_data_coverage=0.6              # 60% coverage
    )
    
    features = BehavioralActivationFeatures(timing_config=config)
    
    print(f"  Configuration: {config.analysis_window_days} days, {config.time_resolution_minutes}min resolution")
    print(f"  Variance metric: {config.variance_metric}, Smoothing: {config.smooth_activity_data}")
    
    # Step 2: Data Generation
    print("\nStep 2: Data Generation")
    accel_data = create_regular_activity_data(days=7, samples_per_hour=60)
    
    print(f"  Generated {len(accel_data['timestamp'])} accelerometer readings")
    print(f"  Time span: {(accel_data['timestamp'][-1] - accel_data['timestamp'][0]).days + 1} days")
    print(f"  Sampling rate: {len(accel_data['timestamp']) / ((accel_data['timestamp'][-1] - accel_data['timestamp'][0]).total_seconds() / 3600):.1f} samples/hour")
    
    # Step 3: Feature Extraction
    print("\nStep 3: Activity Timing Variance Extraction")
    result = features.activity_timing_variance(accel_data)
    
    print(f"  ✓ Extraction completed successfully")
    print(f"  ✓ Found {len(result['daily_variances'])} daily patterns")
    print(f"  ✓ Calculated variance: {result['weekly_variance']:.4f}")
    print(f"  ✓ Peak activity hour: {result['activity_timing_variance']['peak_activity_hour']}:00")
    
    # Step 4: Quality Assessment
    print("\nStep 4: Quality Assessment")
    quality = result['quality_metrics']
    
    print(f"  Data quality score: {quality['overall_quality']:.3f}/1.0")
    print(f"  Coverage ratio: {quality['coverage_ratio']:.1%}")
    print(f"  Sampling rate: {quality['sampling_rate_hz']:.1f} Hz")
    print(f"  Data completeness: {quality['data_completeness']:.1%}")
    print(f"  Temporal consistency: {quality['temporal_consistency']:.3f}")
    
    if quality['overall_quality'] > 0.7:
        quality_assessment = "High quality timing data"
    elif quality['overall_quality'] > 0.4:
        quality_assessment = "Acceptable quality timing data"
    else:
        quality_assessment = "Low quality timing data - interpret with caution"
    
    print(f"  Assessment: {quality_assessment}")
    
    # Step 5: Results Interpretation
    print("\nStep 5: Results Interpretation")
    variance = result['activity_timing_variance']
    
    print(f"  Weekly variance: {variance['weekly_variance']:.4f}")
    print(f"  Peak activity hour: {variance['peak_activity_hour']}:00")
    print(f"  Variance stability: {variance['variance_stability']:.3f}")
    print(f"  Mean hourly variance: {variance['mean_hourly_variance']:.4f}")
    print(f"  Max hourly variance: {variance['max_hourly_variance']:.4f}")
    
    # Behavioral interpretation
    if variance['weekly_variance'] < 0.02:
        behavioral_pattern = "Highly regular - consistent daily routine"
        activation_level = "Stable behavioral activation"
    elif variance['weekly_variance'] < 0.05:
        behavioral_pattern = "Moderately regular - some daily flexibility"
        activation_level = "Moderate behavioral activation stability"
    else:
        behavioral_pattern = "Irregular - variable daily patterns"
        activation_level = "Variable behavioral activation"
    
    print(f"  Behavioral pattern: {behavioral_pattern}")
    print(f"  Activation level: {activation_level}")
    print(f"  Stability score: {variance['variance_stability']:.3f}")
    
    # Step 6: Hourly Pattern Analysis
    print("\nStep 6: Hourly Pattern Analysis")
    hourly_variances = variance['hourly_patterns']['hourly_variances']
    
    print(f"  Top 5 most variable hours:")
    sorted_hours = sorted(enumerate(hourly_variances), key=lambda x: x[1], reverse=True)
    
    for i, (hour, var) in enumerate(sorted_hours[:5]):
        print(f"    {i+1}. {hour:02d}:00 - variance: {var:.4f}")
    
    # Step 7: Day Type Analysis
    print("\nStep 7: Day Type Analysis")
    if 'variance_by_day_type' in result and result['variance_by_day_type']:
        variance_by_type = result['variance_by_day_type']
        
        print(f"  Weekday variance: {variance_by_type.get('weekday', 'N/A'):.4f}")
        print(f"  Weekend variance: {variance_by_type.get('weekend', 'N/A'):.4f}")
        
        if 'weekday_weekend_difference' in variance_by_type:
            diff = variance_by_type['weekday_weekend_difference']
            if abs(diff) < 0.01:
                pattern_note = "Similar weekday/weekend patterns"
            elif diff > 0:
                pattern_note = "More variable weekdays"
            else:
                pattern_note = "More variable weekends"
            
            print(f"  Pattern: {pattern_note} (difference: {abs(diff):.4f})")
    
    # Step 8: Clinical Interpretation
    print("\nStep 8: Clinical Interpretation")
    
    if variance['weekly_variance'] > 0.08:
        clinical_note = "High timing variability may indicate disrupted routines or irregular sleep patterns"
    elif variance['weekly_variance'] > 0.04:
        clinical_note = "Moderate timing variability suggests some schedule flexibility or lifestyle changes"
    elif variance['weekly_variance'] > 0.02:
        clinical_note = "Low timing variability indicates stable daily routines and consistent patterns"
    else:
        clinical_note = "Very low timing variability suggests highly structured routines, possibly rigid"
    
    print(f"  Clinical note: {clinical_note}")
    
    # Step 9: Processing Summary
    print("\nStep 9: Processing Summary")
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
    """Run all activity timing variance examples."""
    print("Psyconstruct Activity Timing Variance Feature Examples")
    print("=" * 60)
    
    example_basic_activity_timing_variance()
    example_custom_configuration()
    example_pattern_comparison()
    example_variance_metrics_comparison()
    example_shift_worker_analysis()
    example_time_resolution_analysis()
    example_quality_assessment()
    example_variance_interpretation()
    example_complete_analysis_workflow()
    
    print("All activity timing variance examples completed successfully!")
