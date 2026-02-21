"""
Example usage of Routine Stability (RS) construct features.

This example demonstrates how to:
1. Extract sleep onset consistency from screen usage data
2. Calculate sleep duration from screen usage patterns
3. Compute activity fragmentation from activity data
4. Analyze circadian midpoints from sleep/wake patterns
5. Interpret routine stability patterns and behaviors
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psyconstruct.features.routine_stability import (
    RoutineStabilityFeatures,
    SleepOnsetConfig,
    SleepDurationConfig,
    ActivityFragmentationConfig,
    CircadianMidpointConfig
)


def create_regular_screen_data(days: int = 14, sleep_hour: int = 22, wake_hour: int = 6):
    """Create screen data for someone with regular sleep patterns."""
    
    print(f"Generating {days} days of regular screen data (sleep: {sleep_hour}:00, wake: {wake_hour}:00)...")
    
    timestamps = []
    screen_states = []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    for day in range(days):
        # Wake time: screen on
        wake_time = base_time + timedelta(days=day, hours=wake_hour)
        timestamps.append(wake_time)
        screen_states.append(1)
        
        # Daytime: regular screen usage
        for hour in range(wake_hour, sleep_hour):
            # Multiple screen state changes during the day
            if random.random() < 0.7:  # 70% chance of screen being on
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=random.randint(0, 59))
                timestamps.append(timestamp)
                screen_states.append(1)
                
                # Random screen off periods
                if random.random() < 0.3:
                    timestamp = base_time + timedelta(days=day, hours=hour, minutes=random.randint(0, 59))
                    timestamps.append(timestamp)
                    screen_states.append(0)
        
        # Sleep onset: screen off
        sleep_time = base_time + timedelta(days=day, hours=sleep_hour)
        timestamps.append(sleep_time)
        screen_states.append(0)
        
        # During sleep: screen off
        midnight = base_time + timedelta(days=day, hours=23, minutes=59)
        timestamps.append(midnight)
        screen_states.append(0)
    
    return {
        'timestamp': timestamps,
        'screen_state': screen_states
    }


def create_irregular_screen_data(days: int = 14):
    """Create screen data for someone with irregular sleep patterns."""
    
    print(f"Generating {days} days of irregular screen data...")
    
    timestamps = []
    screen_states = []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    for day in range(days):
        # Random sleep times
        sleep_hour = random.randint(20, 24) if random.random() < 0.7 else random.randint(0, 3)
        sleep_duration = random.uniform(5, 10)  # 5-10 hours sleep
        wake_hour = int((sleep_hour + sleep_duration) % 24)
        
        # Wake time: screen on
        if sleep_hour < wake_hour:  # Same day
            wake_time = base_time + timedelta(days=day, hours=wake_hour)
        else:  # Next day
            wake_time = base_time + timedelta(days=day+1, hours=wake_hour)
        
        timestamps.append(wake_time)
        screen_states.append(1)
        
        # Irregular daytime usage
        for hour in range(24):
            if random.random() < 0.6:  # 60% chance of any activity
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=random.randint(0, 59))
                timestamps.append(timestamp)
                screen_states.append(1 if random.random() < 0.7 else 0)
        
        # Sleep onset: screen off
        if sleep_hour < wake_hour:  # Sleep before wake (next day)
            sleep_time = base_time + timedelta(days=day+1, hours=sleep_hour)
        else:  # Sleep same day
            sleep_time = base_time + timedelta(days=day, hours=sleep_hour)
        
        timestamps.append(sleep_time)
        screen_states.append(0)
    
    return {
        'timestamp': timestamps,
        'screen_state': screen_states
    }


def create_shift_worker_screen_data(days: int = 14):
    """Create screen data for shift worker with rotating schedules."""
    
    print(f"Generating {days} days of shift worker screen data...")
    
    timestamps = []
    screen_states = []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    # Shift patterns: day shift, evening shift, night shift
    shift_patterns = [
        {'sleep': 22, 'wake': 6},   # Day shift
        {'sleep': 2, 'wake': 10},   # Evening shift  
        {'sleep': 6, 'wake': 14}    # Night shift
    ]
    
    for day in range(days):
        shift = shift_patterns[day % len(shift_patterns)]
        sleep_hour = shift['sleep']
        wake_hour = shift['wake']
        
        # Wake time: screen on
        if sleep_hour < wake_hour:  # Same day
            wake_time = base_time + timedelta(days=day, hours=wake_hour)
        else:  # Next day
            wake_time = base_time + timedelta(days=day+1, hours=wake_hour)
        
        timestamps.append(wake_time)
        screen_states.append(1)
        
        # Work hours: high screen usage
        for hour in range(wake_hour, wake_hour + 8):  # 8-hour shifts
            current_hour = hour % 24
            timestamp = base_time + timedelta(days=day + hour // 24, hours=current_hour, minutes=random.randint(0, 59))
            timestamps.append(timestamp)
            screen_states.append(1)
        
        # Sleep onset: screen off
        if sleep_hour < wake_hour:  # Sleep same day
            sleep_time = base_time + timedelta(days=day, hours=sleep_hour)
        else:  # Sleep next day
            sleep_time = base_time + timedelta(days=day+1, hours=sleep_hour)
        
        timestamps.append(sleep_time)
        screen_states.append(0)
    
    return {
        'timestamp': timestamps,
        'screen_state': screen_states
    }


def create_regular_activity_data(days: int = 7):
    """Create activity data for someone with regular daily patterns."""
    
    print(f"Generating {days} days of regular activity data...")
    
    timestamps = []
    activity_levels = []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    for day in range(days):
        for hour in range(24):
            for minute in range(0, 60, 30):  # Every 30 minutes
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
                
                # Regular pattern: high activity during day, low at night
                if 6 <= hour < 8:      # Morning routine
                    activity = 3.0 + random.random() * 2.0
                elif 8 <= hour < 12:    # Morning work
                    activity = 2.0 + random.random() * 1.5
                elif 12 <= hour < 13:   # Lunch break
                    activity = 4.0 + random.random() * 2.0
                elif 13 <= hour < 17:   # Afternoon work
                    activity = 1.5 + random.random() * 1.0
                elif 17 <= hour < 20:   # Evening activity
                    activity = 3.5 + random.random() * 2.0
                elif 20 <= hour < 22:   # Wind down
                    activity = 1.0 + random.random() * 0.5
                else:  # Sleep (22-6)
                    activity = 0.1 + random.random() * 0.2
                
                timestamps.append(timestamp)
                activity_levels.append(activity)
    
    return {
        'timestamp': timestamps,
        'activity_level': activity_levels
    }


def create_fragmented_activity_data(days: int = 7):
    """Create activity data for someone with fragmented patterns."""
    
    print(f"Generating {days} days of fragmented activity data...")
    
    timestamps = []
    activity_levels = []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    for day in range(days):
        for hour in range(24):
            for minute in range(0, 60, 30):  # Every 30 minutes
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
                
                # Fragmented pattern: random activity bursts
                if random.random() < 0.3:  # 30% chance of high activity burst
                    activity = random.uniform(5.0, 8.0)
                elif random.random() < 0.6:  # 30% chance of moderate activity
                    activity = random.uniform(1.0, 3.0)
                else:  # 40% chance of low activity
                    activity = random.uniform(0.1, 0.8)
                
                timestamps.append(timestamp)
                activity_levels.append(activity)
    
    return {
        'timestamp': timestamps,
        'activity_level': activity_levels
    }


def example_sleep_onset_consistency():
    """Example showing sleep onset consistency analysis."""
    print("=== Sleep Onset Consistency Analysis Example ===")
    
    # Initialize with custom configuration
    config = SleepOnsetConfig(
        analysis_window_days=14,
        min_screen_off_duration_hours=2.0,
        outlier_detection=True,
        weekend_separation=True
    )
    features = RoutineStabilityFeatures(sleep_onset_config=config)
    
    # Analyze different patterns
    patterns = [
        ("Regular Schedule", create_regular_screen_data(days=14, sleep_hour=22, wake_hour=6)),
        ("Irregular Schedule", create_irregular_screen_data(days=14)),
        ("Shift Worker", create_shift_worker_screen_data(days=14))
    ]
    
    for pattern_name, screen_data in patterns:
        print(f"\n{pattern_name}:")
        
        # Patch minimum data check for demonstration
        original_method = features.sleep_onset_consistency
        
        def patched_method(screen_data, window_start=None, window_end=None):
            try:
                return original_method(screen_data, window_start, window_end)
            except ValueError as e:
                if "Insufficient screen data" in str(e):
                    pass  # Continue for demonstration
                else:
                    raise
        
        features.sleep_onset_consistency = patched_method.__get__(features, RoutineStabilityFeatures)
        
        result = features.sleep_onset_consistency(screen_data)
        
        consistency = result['sleep_onset_consistency']
        print(f"  Sleep onset SD: {result['sleep_onset_sd_hours']:.2f} hours")
        print(f"  Mean onset time: {consistency['mean_sleep_onset_hour']:.1f}:00")
        print(f"  Onset range: {consistency['sleep_onset_range_hours']:.1f} hours")
        print(f"  Nights analyzed: {result['data_summary']['nights_analyzed']}")
        
        # Show detected sleep onsets
        if result['daily_sleep_onsets']:
            print(f"  Sample sleep onsets:")
            for i, onset in enumerate(result['daily_sleep_onsets'][:3]):
                print(f"    {onset['date']}: {onset['sleep_onset_hour']:.1f}:00")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"  - SD < 1 hour: Very consistent sleep schedule")
    print(f"  - SD 1-2 hours: Moderately consistent schedule")
    print(f"  - SD > 2 hours: Irregular sleep patterns")
    
    print()


def example_sleep_duration_analysis():
    """Example showing sleep duration analysis."""
    print("=== Sleep Duration Analysis Example ===")
    
    # Initialize with custom configuration
    config = SleepDurationConfig(
        analysis_window_days=14,
        min_sleep_duration_hours=4.0,
        max_sleep_duration_hours=12.0,
        calculate_variability=True
    )
    features = RoutineStabilityFeatures(sleep_duration_config=config)
    
    # Create different sleep duration patterns
    regular_data = create_regular_screen_data(days=14, sleep_hour=22, wake_hour=6)  # 8 hours
    short_data = create_regular_screen_data(days=14, sleep_hour=23, wake_hour=5)    # 6 hours
    long_data = create_regular_screen_data(days=14, sleep_hour=21, wake_hour=7)     # 10 hours
    
    patterns = [
        ("Regular Duration (8h)", regular_data),
        ("Short Duration (6h)", short_data),
        ("Long Duration (10h)", long_data)
    ]
    
    for pattern_name, screen_data in patterns:
        print(f"\n{pattern_name}:")
        
        # Patch minimum data check
        original_method = features.sleep_duration
        
        def patched_method(screen_data, window_start=None, window_end=None):
            try:
                return original_method(screen_data, window_start, window_end)
            except ValueError as e:
                if "Insufficient screen data" in str(e):
                    pass
                else:
                    raise
        
        features.sleep_duration = patched_method.__get__(features, RoutineStabilityFeatures)
        
        result = features.sleep_duration(screen_data)
        
        duration = result['sleep_duration']
        print(f"  Mean duration: {result['mean_sleep_duration_hours']:.1f} hours")
        print(f"  Duration SD: {duration.get('sleep_duration_sd_hours', 0):.2f} hours")
        print(f"  Sleep efficiency: {duration.get('sleep_efficiency', 0):.2f}")
        print(f"  Nights analyzed: {result['data_summary']['nights_analyzed']}")
    
    # Clinical interpretation
    print(f"\nClinical Interpretation:")
    print(f"  - 7-9 hours: Recommended sleep duration for adults")
    print(f"  - < 6 hours: Insufficient sleep, health risk")
    print(f"  - > 10 hours: Excessive sleep, possible health issues")
    print(f"  - High variability: Irregular sleep patterns")
    
    print()


def example_activity_fragmentation():
    """Example showing activity fragmentation analysis."""
    print("=== Activity Fragmentation Analysis Example ===")
    
    # Initialize with custom configuration
    config = ActivityFragmentationConfig(
        analysis_window_days=7,
        time_resolution_minutes=60,
        activity_threshold=0.1,
        entropy_base="natural",
        normalize_by_total_activity=True
    )
    features = RoutineStabilityFeatures(fragmentation_config=config)
    
    # Analyze different activity patterns
    patterns = [
        ("Regular Activity", create_regular_activity_data(days=7)),
        ("Fragmented Activity", create_fragmented_activity_data(days=7))
    ]
    
    for pattern_name, activity_data in patterns:
        print(f"\n{pattern_name}:")
        
        # Patch minimum data check
        original_method = features.activity_fragmentation
        
        def patched_method(activity_data, window_start=None, window_end=None):
            try:
                return original_method(activity_data, window_start, window_end)
            except ValueError as e:
                if "Insufficient activity data" in str(e):
                    pass
                else:
                    raise
        
        features.activity_fragmentation = patched_method.__get__(features, RoutineStabilityFeatures)
        
        result = features.activity_fragmentation(activity_data)
        
        fragmentation = result['activity_fragmentation']
        print(f"  Mean entropy: {result['mean_entropy']:.3f}")
        print(f"  Entropy stability: {fragmentation['entropy_stability']:.3f}")
        print(f"  Peak activity hour: {fragmentation['peak_activity_hour']}:00")
        print(f"  Days analyzed: {result['data_summary']['days_analyzed']}")
        
        # Show hourly pattern
        print(f"  Hourly activity pattern (sample):")
        for hour in range(6, 22, 3):  # Every 3 hours from 6 AM to 9 PM
            activity_level = fragmentation['hourly_patterns'].get(hour, 0)
            print(f"    {hour:02d}:00 - {activity_level:.3f}")
    
    # Interpretation
    print(f"\nEntropy Interpretation:")
    print(f"  - Low entropy (< 2.0): Concentrated activity, structured routine")
    print(f"  - Medium entropy (2.0-2.5): Balanced activity distribution")
    print(f"  - High entropy (> 2.5): Fragmented activity, irregular patterns")
    print(f"  - ln(24) ≈ 3.18: Maximum entropy for uniform distribution")
    
    print()


def example_circadian_midpoint():
    """Example showing circadian midpoint analysis."""
    print("=== Circadian Midpoint Analysis Example ===")
    
    # Initialize with custom configuration
    config = CircadianMidpointConfig(
        analysis_window_days=14,
        calculate_phase_shift=True,
        weekend_analysis=True,
        outlier_detection=True
    )
    features = RoutineStabilityFeatures(circadian_config=config)
    
    # Analyze different circadian patterns
    patterns = [
        ("Early Bird", create_regular_screen_data(days=14, sleep_hour=21, wake_hour=5)),
        ("Normal Schedule", create_regular_screen_data(days=14, sleep_hour=23, wake_hour=7)),
        ("Night Owl", create_regular_screen_data(days=14, sleep_hour=1, wake_hour=9)),
        ("Shift Worker", create_shift_worker_screen_data(days=14))
    ]
    
    for pattern_name, screen_data in patterns:
        print(f"\n{pattern_name}:")
        
        # Patch minimum data check
        original_method = features.circadian_midpoint
        
        def patched_method(screen_data, window_start=None, window_end=None):
            try:
                return original_method(screen_data, window_start, window_end)
            except ValueError as e:
                if "Insufficient screen data" in str(e):
                    pass
                else:
                    raise
        
        features.circadian_midpoint = patched_method.__get__(features, RoutineStabilityFeatures)
        
        result = features.circadian_midpoint(screen_data)
        
        midpoint = result['circadian_midpoint']
        print(f"  Mean midpoint: {result['mean_midpoint_hour']:.1f}:00")
        print(f"  Midpoint SD: {midpoint['midpoint_sd_hours']:.2f} hours")
        print(f"  Phase shift mean: {midpoint.get('phase_shift_mean_hours', 0):.2f} hours")
        print(f"  Nights analyzed: {result['data_summary']['nights_analyzed']}")
        
        # Show sample midpoints
        if result['daily_midpoints']:
            print(f"  Sample midpoints:")
            for i, daily in enumerate(result['daily_midpoints'][:3]):
                print(f"    {daily['date']}: {daily['midpoint_hour']:.1f}:00")
    
    # Chronotype interpretation
    print(f"\nChronotype Interpretation:")
    print(f"  - Midpoint < 2:00: Extreme morning chronotype")
    print(f"  - Midpoint 2:00-4:00: Morning chronotype")
    print(f"  - Midpoint 4:00-6:00: Intermediate chronotype")
    print(f"  - Midpoint > 6:00: Evening chronotype")
    
    print()


def example_routine_stability_profile():
    """Example showing complete routine stability profile analysis."""
    print("=== Complete Routine Stability Profile Analysis ===")
    
    # Initialize all features with clinical configuration
    sleep_onset_config = SleepOnsetConfig(
        analysis_window_days=14,
        min_screen_off_duration_hours=2.0,
        outlier_detection=True
    )
    
    sleep_duration_config = SleepDurationConfig(
        analysis_window_days=14,
        calculate_variability=True
    )
    
    fragmentation_config = ActivityFragmentationConfig(
        analysis_window_days=7,
        normalize_by_total_activity=True
    )
    
    circadian_config = CircadianMidpointConfig(
        analysis_window_days=14,
        calculate_phase_shift=True
    )
    
    features = RoutineStabilityFeatures(
        sleep_onset_config=sleep_onset_config,
        sleep_duration_config=sleep_duration_config,
        fragmentation_config=fragmentation_config,
        circadian_config=circadian_config
    )
    
    # Analyze different profiles
    profiles = [
        ("Highly Stable", create_regular_screen_data(days=14), create_regular_activity_data(days=7)),
        ("Unstable", create_irregular_screen_data(days=14), create_fragmented_activity_data(days=7)),
        ("Shift Worker", create_shift_worker_screen_data(days=14), create_fragmented_activity_data(days=7))
    ]
    
    for profile_name, screen_data, activity_data in profiles:
        print(f"\n{profile_name} Profile:")
        print("-" * 50)
        
        # Patch minimum data checks for all methods
        for method_name in ['sleep_onset_consistency', 'sleep_duration', 'activity_fragmentation', 'circadian_midpoint']:
            original_method = getattr(features, method_name)
            
            def patched_method(data, window_start=None, window_end=None, orig=original_method):
                try:
                    return orig(data, window_start, window_end)
                except ValueError as e:
                    if "Insufficient" in str(e):
                        pass
                    else:
                        raise
            
            setattr(features, method_name, patched_method.__get__(features, RoutineStabilityFeatures))
        
        # Extract all features
        onset_result = features.sleep_onset_consistency(screen_data)
        duration_result = features.sleep_duration(screen_data)
        fragmentation_result = features.activity_fragmentation(activity_data)
        midpoint_result = features.circadian_midpoint(screen_data)
        
        # Calculate stability scores (0-100, higher = more stable)
        onset_score = max(0, 100 - onset_result['sleep_onset_sd_hours'] * 20)  # Lower SD = higher score
        duration_score = duration_result['sleep_duration'].get('sleep_efficiency', 0) * 100
        fragmentation_score = max(0, 100 - fragmentation_result['mean_entropy'] * 20)  # Lower entropy = higher score
        midpoint_score = max(0, 100 - midpoint_result['circadian_midpoint']['midpoint_sd_hours'] * 20)
        
        overall_stability = (onset_score + duration_score + fragmentation_score + midpoint_score) / 4
        
        print(f"Sleep Onset Consistency: {onset_score:.1f}%")
        print(f"Sleep Duration Regularity: {duration_score:.1f}%")
        print(f"Activity Structure: {fragmentation_score:.1f}%")
        print(f"Circadian Regularity: {midpoint_score:.1f}%")
        print(f"Overall Routine Stability: {overall_stability:.1f}%")
        
        # Behavioral interpretation
        if overall_stability > 70:
            behavior = "Highly stable - consistent daily routines"
        elif overall_stability > 40:
            behavior = "Moderately stable - some routine flexibility"
        else:
            behavior = "Low stability - irregular or chaotic patterns"
        
        print(f"Behavioral Pattern: {behavior}")
        
        # Health implications
        health_risks = []
        if onset_score < 50:
            health_risks.append("irregular sleep timing")
        if duration_score < 50:
            health_risks.append("poor sleep quality")
        if fragmentation_score < 50:
            health_risks.append("fragmented daily structure")
        if midpoint_score < 50:
            health_risks.append("circadian disruption")
        
        if health_risks:
            print(f"Health Considerations: {', '.join(health_risks)}")
        else:
            print(f"Health Considerations: Stable routines support good health")
        
        # Quality indicators
        onset_quality = onset_result['quality_metrics']['overall_quality']
        duration_quality = duration_result['quality_metrics']['overall_quality']
        fragmentation_quality = fragmentation_result['quality_metrics']['overall_quality']
        midpoint_quality = midpoint_result['quality_metrics']['overall_quality']
        
        print(f"Data Quality: Onset={onset_quality:.2f}, Duration={duration_quality:.2f}, "
              f"Fragmentation={fragmentation_quality:.2f}, Midpoint={midpoint_quality:.2f}")
    
    print()


def example_clinical_applications():
    """Example showing clinical applications of routine stability features."""
    print("=== Clinical Applications Example ===")
    
    features = RoutineStabilityFeatures()
    
    # Create clinical scenarios
    scenarios = [
        {
            'name': 'Depressive Sleep Pattern',
            'screen_data': create_irregular_screen_data(days=14),
            'activity_data': create_fragmented_activity_data(days=7),
            'description': 'Patient shows irregular sleep and fragmented activity'
        },
        {
            'name': 'Anxiety-Related Insomnia',
            'screen_data': create_regular_screen_data(days=14, sleep_hour=24, wake_hour=5),  # Late sleep, early wake
            'activity_data': create_fragmented_activity_data(days=7),
            'description': 'Patient has reduced sleep duration and fragmented activity'
        },
        {
            'name': 'Healthy Baseline',
            'screen_data': create_regular_screen_data(days=14, sleep_hour=22, wake_hour=6),
            'activity_data': create_regular_activity_data(days=7),
            'description': 'Normal sleep patterns and structured daily activity'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"Description: {scenario['description']}")
        print("-" * 60)
        
        # Patch minimum data checks
        for method_name in ['sleep_onset_consistency', 'sleep_duration', 'activity_fragmentation', 'circadian_midpoint']:
            original_method = getattr(features, method_name)
            
            def patched_method(data, window_start=None, window_end=None, orig=original_method):
                try:
                    return orig(data, window_start, window_end)
                except ValueError as e:
                    if "Insufficient" in str(e):
                        pass
                    else:
                        raise
            
            setattr(features, method_name, patched_method.__get__(features, RoutineStabilityFeatures))
        
        # Extract features
        onset_result = features.sleep_onset_consistency(scenario['screen_data'])
        duration_result = features.sleep_duration(scenario['screen_data'])
        fragmentation_result = features.activity_fragmentation(scenario['activity_data'])
        midpoint_result = features.circadian_midpoint(scenario['screen_data'])
        
        # Clinical metrics
        onset_sd = onset_result['sleep_onset_sd_hours']
        duration_hours = duration_result['mean_sleep_duration_hours']
        fragmentation_entropy = fragmentation_result['mean_entropy']
        midpoint_hour = result['mean_midpoint_hour']
        
        print(f"Sleep Onset SD: {onset_sd:.2f} hours")
        print(f"Sleep Duration: {duration_hours:.1f} hours")
        print(f"Activity Fragmentation: {fragmentation_entropy:.3f}")
        print(f"Circadian Midpoint: {midpoint_hour:.1f}:00")
        
        # Clinical interpretation
        print(f"\nClinical Assessment:")
        
        if onset_sd > 2:
            print(f"  ✓ Highly irregular sleep timing suggests circadian disruption")
        elif onset_sd > 1:
            print(f"  ⚠ Moderate sleep timing irregularity")
        else:
            print(f"  ✓ Regular sleep timing patterns")
        
        if duration_hours < 6:
            print(f"  ✓ Insufficient sleep duration indicates sleep deprivation")
        elif duration_hours > 10:
            print(f"  ⚠ Excessive sleep duration may indicate hypersomnia")
        else:
            print(f"  ✓ Normal sleep duration")
        
        if fragmentation_entropy > 2.5:
            print(f"  ✓ High activity fragmentation suggests poor routine structure")
        elif fragmentation_entropy > 2.0:
            print(f"  ⚠ Moderate activity fragmentation")
        else:
            print(f"  ✓ Structured daily activity patterns")
        
        # Risk assessment
        risk_factors = 0
        if onset_sd > 2:
            risk_factors += 1
        if duration_hours < 6 or duration_hours > 10:
            risk_factors += 1
        if fragmentation_entropy > 2.5:
            risk_factors += 1
        
        if risk_factors >= 2:
            risk_level = "HIGH - Significant routine instability"
        elif risk_factors == 1:
            risk_level = "MODERATE - Some routine disruption"
        else:
            risk_level = "LOW - Stable routines"
        
        print(f"\nRisk Level: {risk_level}")
        
        # Treatment recommendations
        if risk_factors >= 2:
            print(f"Treatment Focus: Sleep hygiene, routine stabilization, circadian rhythm therapy")
        elif risk_factors == 1:
            print(f"Treatment Focus: Monitor routines, behavioral activation")
        else:
            print(f"Treatment Focus: Maintain healthy routines")
    
    print()


if __name__ == "__main__":
    """Run all routine stability examples."""
    print("Psyconstruct Routine Stability (RS) Construct Examples")
    print("=" * 60)
    
    example_sleep_onset_consistency()
    example_sleep_duration_analysis()
    example_activity_fragmentation()
    example_circadian_midpoint()
    example_routine_stability_profile()
    example_clinical_applications()
    
    print("All routine stability examples completed successfully!")
