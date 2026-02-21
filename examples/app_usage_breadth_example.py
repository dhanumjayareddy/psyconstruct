"""
Example usage of app usage breadth feature.

This example demonstrates how to:
1. Extract app usage breadth from app usage logs
2. Configure entropy calculation and filtering parameters
3. Interpret app usage breadth results
4. Handle different app usage data quality scenarios
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
    AppUsageBreadthConfig
)


def create_realistic_app_usage_data(days: int = 7, sessions_per_day: int = 30):
    """Create realistic app usage data with daily patterns."""
    
    print(f"Generating {days} days of app usage data with {sessions_per_day} sessions per day...")
    
    timestamps = []
    app_names = []
    durations = []
    
    # Define app categories with usage patterns
    apps = {
        'Instagram': {
            'frequency': 0.25,  # 25% of sessions
            'avg_duration': 120,  # 2 minutes average
            'peak_hours': [12, 13, 19, 20, 21],  # Lunch and evening
            'category': 'social'
        },
        'Facebook': {
            'frequency': 0.20,
            'avg_duration': 180,
            'peak_hours': [8, 9, 12, 13, 20, 21],
            'category': 'social'
        },
        'WhatsApp': {
            'frequency': 0.20,
            'avg_duration': 60,
            'peak_hours': [8, 9, 12, 13, 17, 18, 20, 21],
            'category': 'communication'
        },
        'YouTube': {
            'frequency': 0.15,
            'avg_duration': 300,
            'peak_hours': [19, 20, 21, 22],
            'category': 'entertainment'
        },
        'Gmail': {
            'frequency': 0.10,
            'avg_duration': 90,
            'peak_hours': [8, 9, 12, 13, 17, 18],
            'category': 'productivity'
        },
        'Spotify': {
            'frequency': 0.10,
            'avg_duration': 600,
            'peak_hours': [8, 9, 17, 18, 19],
            'category': 'entertainment'
        }
    }
    
    base_time = datetime(2026, 2, 21, 6, 0, 0)  # Start at 6 AM
    
    for day in range(days):
        current_day_offset = day * 24 * 60  # minutes
        
        for session in range(sessions_per_day):
            # Choose app based on frequency
            rand_val = random.random()
            cumulative = 0
            selected_app = None
            
            for app, props in apps.items():
                cumulative += props['frequency']
                if rand_val <= cumulative:
                    selected_app = app
                    break
            
            if selected_app is None:
                selected_app = list(apps.keys())[0]
            
            # Get app properties
            app_props = apps[selected_app]
            
            # Generate duration with variation
            base_duration = app_props['avg_duration']
            duration = base_duration + random.gauss(0, base_duration * 0.3)  # 30% std deviation
            duration = max(30, duration)  # Minimum 30 seconds
            
            # Generate timestamp with peak hour preference
            hour = random.choices(
                app_props['peak_hours'] + list(range(6, 24)),  # Peak hours + all hours
                weights=[0.7] * len(app_props['peak_hours']) + [0.3] * (24 - 6),  # Higher weight for peak hours
                k=1
            )[0]
            
            minute = random.randint(0, 59)
            total_minute = current_day_offset + hour * 60 + minute
            timestamp = base_time + timedelta(minutes=total_minute)
            
            timestamps.append(timestamp)
            app_names.append(selected_app)
            durations.append(int(duration))
    
    return {
        'timestamp': timestamps,
        'app_name': app_names,
        'duration_seconds': durations
    }


def create_high_diversity_data():
    """Create app usage data with high diversity (many different apps)."""
    
    print("Creating high diversity app usage data (many different apps)...")
    
    # Many diverse apps
    diverse_apps = [
        'Instagram', 'Facebook', 'WhatsApp', 'YouTube', 'Gmail', 'Spotify',
        'Twitter', 'LinkedIn', 'Reddit', 'TikTok', 'Snapchat', 'Telegram',
        'Netflix', 'Disney+', 'Amazon', 'eBay', 'Uber', 'Maps', 'Weather'
    ]
    
    timestamps = []
    app_names = []
    durations = []
    
    base_time = datetime(2026, 2, 21, 8, 0, 0)
    
    for day in range(7):
        for session in range(25):  # 25 sessions per day
            # Use many different apps
            app_idx = random.randint(0, len(diverse_apps) - 1)
            app_name = diverse_apps[app_idx]
            
            # Random duration between 30 seconds and 10 minutes
            duration = random.randint(30, 600)
            
            # Random timestamp throughout the day
            hour = random.randint(8, 23)
            minute = random.randint(0, 59)
            timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
            
            timestamps.append(timestamp)
            app_names.append(app_name)
            durations.append(duration)
    
    return {
        'timestamp': timestamps,
        'app_name': app_names,
        'duration_seconds': durations
    }


def create_low_diversity_data():
    """Create app usage data with low diversity (mostly one dominant app)."""
    
    print("Creating low diversity app usage data (dominant app usage)...")
    
    timestamps = []
    app_names = []
    durations = []
    
    base_time = datetime(2026, 2, 21, 8, 0, 0)
    
    for day in range(7):
        for session in range(30):  # 30 sessions per day
            # 80% Instagram, 20% other apps
            if random.random() < 0.8:
                app_name = 'Instagram'
                duration = random.randint(120, 300)  # 2-5 minutes
            else:
                other_apps = ['Facebook', 'WhatsApp', 'YouTube']
                app_name = random.choice(other_apps)
                duration = random.randint(60, 180)  # 1-3 minutes
            
            # Random timestamp
            hour = random.randint(8, 23)
            minute = random.randint(0, 59)
            timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
            
            timestamps.append(timestamp)
            app_names.append(app_name)
            durations.append(duration)
    
    return {
        'timestamp': timestamps,
        'app_name': app_names,
        'duration_seconds': durations
    }


def example_basic_app_usage_breadth():
    """Example showing basic app usage breadth extraction."""
    print("=== Basic App Usage Breadth Example ===")
    
    # Initialize with default configuration
    features = BehavioralActivationFeatures()
    
    # Create sample app usage data
    app_data = create_realistic_app_usage_data(days=7, sessions_per_day=25)
    
    print(f"\nInput data summary:")
    print(f"  Total app sessions: {len(app_data['timestamp'])}")
    print(f"  Time span: {(app_data['timestamp'][-1] - app_data['timestamp'][0]).days} days")
    print(f"  Date range: {app_data['timestamp'][0].date()} to {app_data['timestamp'][-1].date()}")
    print(f"  Unique apps in raw data: {len(set(app_data['app_name']))}")
    
    # Extract app usage breadth
    result = features.app_usage_breadth(app_data)
    
    print(f"\nApp Usage Breadth Results:")
    breadth = result['app_usage_breadth']
    print(f"  Weekly entropy: {breadth['weekly_entropy']:.3f} bits")
    print(f"  Unique apps: {breadth['unique_apps']}")
    print(f"  Total sessions: {breadth['total_sessions']}")
    print(f"  Total usage time: {breadth['total_usage_time']:.0f} seconds ({breadth['total_usage_time']/3600:.1f} hours)")
    print(f"  Dominant app: {breadth['dominant_app']}")
    print(f"  Max possible entropy: {breadth['max_possible_entropy']:.3f} bits")
    print(f"  Entropy per app: {breadth['entropy_per_app']:.3f} bits/app")
    
    # Show app usage patterns
    print(f"\nApp Usage Patterns:")
    sorted_apps = sorted(
        breadth['app_usage_patterns'].items(), 
        key=lambda x: x[1]['usage_probability'], 
        reverse=True
    )
    
    for app, pattern in sorted_apps:
        print(f"  {app}:")
        print(f"    Usage probability: {pattern['usage_probability']:.3f}")
        print(f"    Session count: {pattern['session_count']}")
        print(f"    Total duration: {pattern['total_duration_seconds']:.0f} seconds")
        print(f"    Avg session duration: {pattern['avg_session_duration_seconds']:.0f} seconds")
    
    # Show quality metrics
    quality = result['quality_metrics']
    print(f"\nData Quality Metrics:")
    print(f"  Overall quality: {quality['overall_quality']:.3f}")
    print(f"  Coverage ratio: {quality['coverage_ratio']:.3f}")
    print(f"  Sessions per day: {quality['sessions_per_day']:.1f}")
    print(f"  Unique apps: {quality['unique_apps']}")
    print(f"  Usage consistency: {quality['usage_consistency']:.3f}")
    
    print()


def example_custom_configuration():
    """Example showing custom configuration for app usage breadth."""
    print("=== Custom Configuration Example ===")
    
    # Create custom configuration
    custom_config = AppUsageBreadthConfig(
        analysis_window_days=14,         # 2-week analysis
        min_usage_duration_seconds=60.0, # Minimum 1 minute sessions
        min_app_sessions=5,              # Minimum 5 sessions per app
        categorize_apps=False,           # Don't categorize apps
        exclude_system_apps=False,       # Include system apps
        entropy_base=10.0,               # Base-10 entropy
        include_duration_weighting=False, # Weight by sessions, not duration
        min_total_sessions=20,           # Lower minimum sessions
        min_active_days=5,               # Require 5 active days
        normalize_by_total_time=False     # Don't normalize by total time
    )
    
    features = BehavioralActivationFeatures(app_usage_config=custom_config)
    
    print("Custom configuration parameters:")
    print(f"  Analysis window: {custom_config.analysis_window_days} days")
    print(f"  Minimum usage duration: {custom_config.min_usage_duration_seconds} seconds")
    print(f"  Minimum app sessions: {custom_config.min_app_sessions}")
    print(f"  Entropy base: {custom_config.entropy_base}")
    print(f"  Duration weighting: {custom_config.include_duration_weighting}")
    print(f"  Exclude system apps: {custom_config.exclude_system_apps}")
    
    # Create app usage data
    app_data = create_realistic_app_usage_data(days=14, sessions_per_day=20)
    
    # Extract with custom configuration
    result = features.app_usage_breadth(app_data)
    
    print(f"\nResults with custom configuration:")
    print(f"  Weekly entropy: {result['weekly_entropy']:.3f}")
    print(f"  Unique apps: {result['unique_apps']}")
    print(f"  Total sessions: {result['total_sessions']}")
    print(f"  Processing used {len(result['app_usage_patterns'])} apps")
    
    print()


def example_diversity_comparison():
    """Example comparing high vs low app usage diversity."""
    print("=== App Usage Diversity Comparison Example ===")
    
    # Initialize with lenient configuration for testing
    config = AppUsageBreadthConfig(
        min_total_sessions=20,
        min_app_sessions=2,
        min_usage_duration_seconds=30.0
    )
    features = BehavioralActivationFeatures(app_usage_config=config)
    
    # Test high diversity data
    print("Testing high diversity pattern...")
    high_diversity_data = create_high_diversity_data()
    high_result = features.app_usage_breadth(high_diversity_data)
    
    print(f"  High diversity entropy: {high_result['weekly_entropy']:.3f}")
    print(f"  High diversity apps: {high_result['unique_apps']}")
    print(f"  High diversity sessions: {high_result['total_sessions']}")
    
    # Test low diversity data
    print("\nTesting low diversity pattern...")
    low_diversity_data = create_low_diversity_data()
    low_result = features.app_usage_breadth(low_diversity_data)
    
    print(f"  Low diversity entropy: {low_result['weekly_entropy']:.3f}")
    print(f"  Low diversity apps: {low_result['unique_apps']}")
    print(f"  Low diversity sessions: {low_result['total_sessions']}")
    
    # Comparison
    print(f"\nDiversity comparison:")
    entropy_diff = high_result['weekly_entropy'] - low_result['weekly_entropy']
    print(f"  Entropy difference: {entropy_diff:.3f} bits")
    if low_result['weekly_entropy'] > 0:
        print(f"  High diversity has {entropy_diff/low_result['weekly_entropy']*100:.1f}% more entropy")
    print(f"  App difference: {high_result['unique_apps'] - low_result['unique_apps']}")
    
    # Show dominant apps
    print(f"\nDominant app comparison:")
    print(f"  High diversity dominant: {high_result['app_usage_breadth']['dominant_app']}")
    print(f"  Low diversity dominant: {low_result['app_usage_breadth']['dominant_app']}")
    
    print()


def example_weighting_comparison():
    """Example comparing duration weighting vs session counting."""
    print("=== Weighting Method Comparison Example ===")
    
    # Create data where weighting makes a difference
    app_data = {
        'timestamp': [datetime(2026, 2, 21, 8, 0, 0) + timedelta(hours=i) for i in range(20)],
        'app_name': ['Instagram'] * 5 + ['Facebook'] * 10 + ['YouTube'] * 5,
        'duration_seconds': [600] * 5 + [120] * 10 + [180] * 5  # Instagram: long but few, Facebook: many but short
    }
    
    print("Test data:")
    print("  Instagram: 5 sessions, 10 minutes each (50 minutes total)")
    print("  Facebook: 10 sessions, 2 minutes each (20 minutes total)")
    print("  YouTube: 5 sessions, 3 minutes each (15 minutes total)")
    
    # Test with session weighting
    print("\nSession-based weighting:")
    config_sessions = AppUsageBreadthConfig(
        min_total_sessions=10,
        min_app_sessions=2,
        include_duration_weighting=False  # Weight by sessions
    )
    features_sessions = BehavioralActivationFeatures(app_usage_config=config_sessions)
    result_sessions = features_sessions.app_usage_breadth(app_data)
    
    print(f"  Weekly entropy: {result_sessions['weekly_entropy']:.3f}")
    print(f"  Dominant app: {result_sessions['app_usage_breadth']['dominant_app']}")
    
    # Test with duration weighting
    print("\nDuration-based weighting:")
    config_duration = AppUsageBreadthConfig(
        min_total_sessions=10,
        min_app_sessions=2,
        include_duration_weighting=True  # Weight by duration
    )
    features_duration = BehavioralActivationFeatures(app_usage_config=config_duration)
    result_duration = features_duration.app_usage_breadth(app_data)
    
    print(f"  Weekly entropy: {result_duration['weekly_entropy']:.3f}")
    print(f"  Dominant app: {result_duration['app_usage_breadth']['dominant_app']}")
    
    print(f"\nComparison:")
    print(f"  Session weighting favors: {result_sessions['app_usage_breadth']['dominant_app']} (most sessions)")
    print(f"  Duration weighting favors: {result_duration['app_usage_breadth']['dominant_app']} (most time)")
    
    print()


def example_system_app_filtering():
    """Example showing system app filtering."""
    print("=== System App Filtering Example ===")
    
    # Create data with system apps
    app_data = {
        'timestamp': [datetime(2026, 2, 21, 8, 0, 0) + timedelta(hours=i) for i in range(15)],
        'app_name': [
            'Instagram', 'Facebook', 'System', 'Settings', 'WhatsApp',
            'Android', 'YouTube', 'Messages', 'Gmail', 'Phone',
            'Instagram', 'Facebook', 'WhatsApp', 'YouTube', 'Gmail'
        ],
        'duration_seconds': [120] * 15
    }
    
    print("Test data includes:")
    print("  User apps: Instagram, Facebook, WhatsApp, YouTube, Gmail")
    print("  System apps: System, Settings, Android, Messages, Phone")
    
    # Test with system app exclusion
    print("\nWith system app exclusion:")
    config_exclude = AppUsageBreadthConfig(
        min_total_sessions=5,
        min_app_sessions=1,
        exclude_system_apps=True
    )
    features_exclude = BehavioralActivationFeatures(app_usage_config=config_exclude)
    result_exclude = features_exclude.app_usage_breadth(app_data)
    
    print(f"  Unique apps: {result_exclude['unique_apps']}")
    print(f"  Apps included: {list(result_exclude['app_usage_patterns'].keys())}")
    
    # Test with system app inclusion
    print("\nWith system app inclusion:")
    config_include = AppUsageBreadthConfig(
        min_total_sessions=5,
        min_app_sessions=1,
        exclude_system_apps=False
    )
    features_include = BehavioralActivationFeatures(app_usage_config=config_include)
    result_include = features_include.app_usage_breadth(app_data)
    
    print(f"  Unique apps: {result_include['unique_apps']}")
    print(f"  Apps included: {list(result_include['app_usage_patterns'].keys())}")
    
    print()


def example_quality_assessment():
    """Example showing app usage quality assessment."""
    print("=== App Usage Quality Assessment Example ===")
    
    features = BehavioralActivationFeatures()
    
    # Test different quality scenarios
    scenarios = [
        ("High Quality", create_realistic_app_usage_data(days=3, sessions_per_day=40)),
        ("Medium Quality", create_realistic_app_usage_data(days=3, sessions_per_day=20)),
        ("Low Quality", create_realistic_app_usage_data(days=3, sessions_per_day=5))
    ]
    
    for scenario_name, app_data in scenarios:
        print(f"\n{scenario_name} App Usage Data:")
        
        # Assess quality
        quality = features._assess_app_usage_quality(
            app_data['timestamp'],
            app_data['app_name'],
            app_data['duration_seconds']
        )
        
        print(f"  Sessions: {len(app_data['timestamp'])}")
        print(f"  Overall quality: {quality['overall_quality']:.3f}")
        print(f"  Coverage ratio: {quality['coverage_ratio']:.3f}")
        print(f"  Sessions per day: {quality['sessions_per_day']:.1f}")
        print(f"  Unique apps: {quality['unique_apps']}")
        print(f"  Usage consistency: {quality['usage_consistency']:.3f}")
        
        # Try extraction with appropriate config
        min_sessions = min(10, len(app_data['timestamp']))
        config = AppUsageBreadthConfig(
            min_total_sessions=min_sessions,
            min_app_sessions=2,
            min_usage_duration_seconds=30.0
        )
        
        features_with_config = BehavioralActivationFeatures(app_usage_config=config)
        
        try:
            result = features_with_config.app_usage_breadth(app_data)
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
        ("Single App", ['Instagram']),
        ("Two Apps", ['Instagram', 'Facebook']),
        ("Three Apps", ['Instagram', 'Facebook', 'WhatsApp']),
        ("Five Apps", ['Instagram', 'Facebook', 'WhatsApp', 'YouTube', 'Gmail'])
    ]
    
    for pattern_name, app_list in patterns:
        # Create app usage data for this pattern
        app_data = {
            'timestamp': [],
            'app_name': [],
            'duration_seconds': []
        }
        
        base_time = datetime(2026, 2, 21, 8, 0, 0)
        
        for day in range(7):
            for session in range(20):
                app_idx = session % len(app_list)
                app_name = app_list[app_idx]
                
                # Vary duration slightly
                duration = 120 + random.randint(-30, 30)
                
                timestamp = base_time + timedelta(days=day, hours=8 + session * 12/20)
                
                app_data['timestamp'].append(timestamp)
                app_data['app_name'].append(app_name)
                app_data['duration_seconds'].append(duration)
        
        # Extract app usage breadth
        config = AppUsageBreadthConfig(
            min_total_sessions=10,
            min_app_sessions=2,
            min_usage_duration_seconds=30.0
        )
        
        features_with_config = BehavioralActivationFeatures(app_usage_config=config)
        result = features_with_config.app_usage_breadth(app_data)
        
        entropy = result['weekly_entropy']
        max_entropy = result['app_usage_breadth']['max_possible_entropy']
        diversity_ratio = entropy / max_entropy if max_entropy > 0 else 0
        
        print(f"\n{pattern_name} Pattern:")
        print(f"  Entropy: {entropy:.3f} bits")
        print(f"  Max possible entropy: {max_entropy:.3f} bits")
        print(f"  Diversity ratio: {diversity_ratio:.3f}")
        
        # Interpretation
        if diversity_ratio < 0.3:
            interpretation = "Very low diversity - highly focused app usage"
        elif diversity_ratio < 0.6:
            interpretation = "Low to moderate diversity - somewhat focused usage"
        elif diversity_ratio < 0.8:
            interpretation = "Moderate to high diversity - balanced app usage"
        else:
            interpretation = "High diversity - varied app usage patterns"
        
        print(f"  Interpretation: {interpretation}")
    
    print()


def example_complete_analysis_workflow():
    """Example showing complete app usage breadth analysis workflow."""
    print("=== Complete App Usage Breadth Analysis Workflow ===")
    
    # Step 1: Configuration
    print("Step 1: Configuration Setup")
    config = AppUsageBreadthConfig(
        analysis_window_days=7,            # Weekly analysis
        min_usage_duration_seconds=60.0,   # 1-minute minimum sessions
        min_app_sessions=3,                # 3 sessions minimum per app
        min_total_sessions=30,             # 30 sessions minimum total
        entropy_base=2.0,                  # Binary entropy (bits)
        categorize_apps=False,             # Don't categorize for now
        exclude_system_apps=True,          # Exclude system apps
        include_duration_weighting=True,   # Weight by usage duration
        min_active_days=5,                 # Require 5 active days
        normalize_by_total_time=False      # Don't normalize
    )
    
    features = BehavioralActivationFeatures(app_usage_config=config)
    
    print(f"  Configuration: {config.analysis_window_days} days, {config.min_usage_duration_seconds}s min duration")
    print(f"  Weighting: {'duration' if config.include_duration_weighting else 'sessions'}")
    
    # Step 2: Data Generation
    print("\nStep 2: Data Generation")
    app_data = create_realistic_app_usage_data(days=7, sessions_per_day=35)
    
    print(f"  Generated {len(app_data['timestamp'])} app sessions")
    print(f"  Time span: {(app_data['timestamp'][-1] - app_data['timestamp'][0]).days + 1} days")
    print(f"  Raw unique apps: {len(set(app_data['app_name']))}")
    
    # Step 3: Feature Extraction
    print("\nStep 3: App Usage Breadth Extraction")
    result = features.app_usage_breadth(app_data)
    
    print(f"  ✓ Extraction completed successfully")
    print(f"  ✓ Found {result['unique_apps']} unique apps after filtering")
    print(f"  ✓ Calculated entropy: {result['weekly_entropy']:.3f} bits")
    print(f"  ✓ Total usage time: {result['total_usage_time']/3600:.1f} hours")
    
    # Step 4: Quality Assessment
    print("\nStep 4: Quality Assessment")
    quality = result['quality_metrics']
    
    print(f"  Data quality score: {quality['overall_quality']:.3f}/1.0")
    print(f"  Coverage ratio: {quality['coverage_ratio']:.1%}")
    print(f"  Sessions per day: {quality['sessions_per_day']:.1f}")
    print(f"  Usage consistency: {quality['usage_consistency']:.3f}")
    
    if quality['overall_quality'] > 0.7:
        quality_assessment = "High quality data"
    elif quality['overall_quality'] > 0.4:
        quality_assessment = "Acceptable quality data"
    else:
        quality_assessment = "Low quality data - interpret with caution"
    
    print(f"  Assessment: {quality_assessment}")
    
    # Step 5: Results Interpretation
    print("\nStep 5: Results Interpretation")
    breadth = result['app_usage_breadth']
    
    print(f"  Weekly entropy: {breadth['weekly_entropy']:.3f} bits")
    print(f"  Unique apps: {breadth['unique_apps']}")
    print(f"  Total sessions: {breadth['total_sessions']}")
    print(f"  Dominant app: {breadth['dominant_app']}")
    print(f"  Max possible entropy: {breadth['max_possible_entropy']:.3f} bits")
    
    # Behavioral interpretation
    entropy_per_app = breadth['weekly_entropy'] / breadth['unique_apps'] if breadth['unique_apps'] > 0 else 0
    
    if breadth['weekly_entropy'] < 1.5:
        behavioral_pattern = "Highly focused - strong app preferences"
        activation_level = "Lower digital activation"
    elif breadth['weekly_entropy'] < 2.5:
        behavioral_pattern = "Moderately focused - some app variety"
        activation_level = "Moderate digital activation"
    else:
        behavioral_pattern = "High variety - diverse app usage"
        activation_level = "Higher digital activation"
    
    print(f"  Behavioral pattern: {behavioral_pattern}")
    print(f"  Activation level: {activation_level}")
    print(f"  Entropy per app: {entropy_per_app:.3f} bits/app")
    
    # Step 6: App Usage Analysis
    print("\nStep 6: App Usage Pattern Analysis")
    patterns = result['app_usage_patterns']
    
    # Sort by usage probability
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['usage_probability'], reverse=True)
    
    print(f"  Top 5 most used apps:")
    for i, (app, pattern) in enumerate(sorted_patterns[:5]):
        percentage = pattern['usage_probability'] * 100
        avg_minutes = pattern['avg_session_duration_seconds'] / 60
        print(f"    {i+1}. {app}: {percentage:.1f}% usage, "
              f"{pattern['session_count']} sessions, {avg_minutes:.1f} min avg")
    
    # Calculate usage distribution
    total_time = sum(p['total_duration_seconds'] for p in patterns.values())
    top_3_time = sum(sorted_patterns[:3][i][1]['total_duration_seconds'] for i in range(min(3, len(sorted_patterns))))
    concentration_ratio = top_3_time / total_time if total_time > 0 else 0
    
    print(f"\n  Usage concentration:")
    print(f"    Top 3 apps account for {concentration_ratio:.1%} of total usage time")
    
    # Step 7: Clinical Interpretation
    print("\nStep 7: Clinical Interpretation")
    
    if breadth['weekly_entropy'] > 3.0:
        clinical_note = "High app diversity suggests good digital engagement and exploration"
    elif breadth['weekly_entropy'] > 2.0:
        clinical_note = "Moderate app diversity indicates balanced digital routine"
    elif breadth['weekly_entropy'] > 1.0:
        clinical_note = "Lower diversity may indicate focused digital interests or routine"
    else:
        clinical_note = "Very low diversity may suggest digital restriction or limited engagement"
    
    print(f"  Clinical note: {clinical_note}")
    
    # Step 8: Processing Summary
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
    """Run all app usage breadth examples."""
    print("Psyconstruct App Usage Breadth Feature Examples")
    print("=" * 60)
    
    example_basic_app_usage_breadth()
    example_custom_configuration()
    example_diversity_comparison()
    example_weighting_comparison()
    example_system_app_filtering()
    example_quality_assessment()
    example_entropy_interpretation()
    example_complete_analysis_workflow()
    
    print("All app usage breadth examples completed successfully!")
