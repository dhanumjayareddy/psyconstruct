"""
Example usage of data harmonization module.

This example demonstrates how to:
1. Harmonize different sensor data types
2. Apply temporal segmentation
3. Handle missing data
4. Track provenance of harmonization operations
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psyconstruct.preprocessing.harmonization import (
    DataHarmonizer,
    HarmonizationConfig
)


def create_sample_gps_data():
    """Create sample GPS data for demonstration."""
    base_time = datetime(2026, 2, 21, 12, 0, 0)
    timestamps = [base_time + timedelta(minutes=i*5) for i in range(20)]
    
    # Simulate movement pattern
    base_lat, base_lon = 40.7128, -74.0060
    data = {
        'timestamp': timestamps,
        'latitude': [base_lat + i*0.0005 for i in range(20)],
        'longitude': [base_lon + i*0.0005 for i in range(20)]
    }
    return data


def create_sample_accelerometer_data():
    """Create sample accelerometer data for demonstration."""
    base_time = datetime(2026, 2, 21, 12, 0, 0)
    timestamps = [base_time + timedelta(seconds=i*30) for i in range(120)]  # 1 hour of data
    
    # Simulate activity pattern
    data = {
        'timestamp': timestamps,
        'x': [0.1 + 0.05 * (i % 20) for i in range(120)],
        'y': [0.2 + 0.03 * (i % 15) for i in range(120)],
        'z': [9.8 + 0.1 * (i % 10) for i in range(120)]
    }
    return data


def create_sample_communication_data():
    """Create sample communication data for demonstration."""
    base_time = datetime(2026, 2, 21, 8, 0, 0)
    
    # Simulate communication events throughout the day
    events = []
    current_time = base_time
    
    # Morning communications
    for i in range(5):
        events.append({
            'timestamp': current_time + timedelta(hours=i*0.5),
            'direction': 'out' if i % 2 == 0 else 'in',
            'contact_id': f'contact_{i % 3 + 1}'
        })
    
    # Afternoon communications
    afternoon_start = base_time + timedelta(hours=4)
    for i in range(3):
        events.append({
            'timestamp': afternoon_start + timedelta(hours=i*1.5),
            'direction': 'incoming' if i % 2 == 0 else 'outgoing',
            'contact_id': f'contact_{i % 2 + 4}'
        })
    
    # Convert to column format
    data = {
        'timestamp': [event['timestamp'] for event in events],
        'direction': [event['direction'] for event in events],
        'contact_id': [event['contact_id'] for event in events]
    }
    return data


def create_sample_screen_state_data():
    """Create sample screen state data for demonstration."""
    base_time = datetime(2026, 2, 21, 20, 0, 0)  # Evening data
    timestamps = [base_time + timedelta(minutes=i*2) for i in range(30)]
    
    # Simulate screen usage pattern
    states = []
    for i in range(30):
        # Simulate periods of screen on/off
        if i < 5:    # First 10 minutes: screen on
            states.append('on')
        elif i < 8:  # Next 6 minutes: screen off
            states.append('off')
        elif i < 15: # Next 14 minutes: screen on
            states.append('on')
        elif i < 18: # Next 6 minutes: screen off
            states.append('off')
        else:        # Final 12 minutes: screen on
            states.append('on')
    
    data = {
        'timestamp': timestamps,
        'state': states
    }
    return data


def example_basic_harmonization():
    """Example showing basic harmonization of all data types."""
    print("=== Basic Data Harmonization Example ===")
    
    # Create harmonizer with default config
    harmonizer = DataHarmonizer()
    
    # Create sample data
    gps_data = create_sample_gps_data()
    accel_data = create_sample_accelerometer_data()
    comm_data = create_sample_communication_data()
    screen_data = create_sample_screen_state_data()
    
    print(f"Original data sizes:")
    print(f"  GPS: {len(gps_data['timestamp'])} records")
    print(f"  Accelerometer: {len(accel_data['timestamp'])} records")
    print(f"  Communication: {len(comm_data['timestamp'])} records")
    print(f"  Screen state: {len(screen_data['timestamp'])} records")
    
    # Harmonize all data types
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore mock implementation warnings
        
        harmonized_gps = harmonizer.harmonize_gps_data(gps_data)
        harmonized_accel = harmonizer.harmonize_accelerometer_data(accel_data)
        harmonized_comm = harmonizer.harmonize_communication_data(comm_data)
        harmonized_screen = harmonizer.harmonize_screen_state_data(screen_data)
    
    print(f"\nHarmonized data sizes:")
    print(f"  GPS: {len(harmonized_gps['timestamp'])} records")
    print(f"  Accelerometer: {len(harmonized_accel['timestamp'])} records")
    print(f"  Communication: {len(harmonized_comm['timestamp'])} records")
    print(f"  Screen state: {len(harmonized_screen['timestamp'])} records")
    
    # Show metadata
    print(f"\nGPS harmonization metadata:")
    gps_meta = harmonized_gps['harmonization_metadata']
    for key, value in gps_meta.items():
        print(f"  {key}: {value}")
    
    print()


def example_temporal_segmentation():
    """Example showing temporal segmentation application."""
    print("=== Temporal Segmentation Example ===")
    
    harmonizer = DataHarmonizer()
    gps_data = create_sample_gps_data()
    
    # Apply temporal segmentation
    segmented_gps = harmonizer.apply_temporal_segmentation(gps_data, 'gps')
    
    flags = segmented_gps['temporal_flags']
    
    print("Temporal segmentation results:")
    print(f"  Weekend flags: {flags['is_weekend'][:5]}...")  # Show first 5
    print(f"  Work hours flags: {flags['is_work_hours'][:5]}...")
    print(f"  Hours of day: {flags['hour_of_day'][:5]}...")
    print(f"  Days of week: {flags['day_of_week'][:5]}...")
    
    # Show statistics
    weekend_count = sum(flags['is_weekend'])
    work_hours_count = sum(flags['is_work_hours'])
    
    print(f"\nTemporal statistics:")
    print(f"  Weekend records: {weekend_count}/{len(flags['is_weekend'])}")
    print(f"  Work hours records: {work_hours_count}/{len(flags['is_work_hours'])}")
    print()


def example_missing_data_handling():
    """Example showing missing data detection and handling."""
    print("=== Missing Data Handling Example ===")
    
    harmonizer = DataHarmonizer()
    
    # Create data with gaps
    base_time = datetime(2026, 2, 21, 12, 0, 0)
    sparse_data = {
        'timestamp': [
            base_time,
            base_time + timedelta(hours=1),  # 1 hour gap
            base_time + timedelta(hours=6),  # 5 hour gap
            base_time + timedelta(hours=12)  # 6 hour gap
        ],
        'latitude': [40.7128, 40.7228, 40.7328, 40.7428],
        'longitude': [-74.0060, -74.0160, -74.0260, -74.0360]
    }
    
    # Apply missing data analysis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flagged_data = harmonizer.apply_missing_data_flags(sparse_data, 'gps')
    
    flags = flagged_data['missing_data_flags']
    
    print("Missing data analysis results:")
    print(f"  Coverage ratio: {flags['coverage_ratio']:.3f}")
    print(f"  Maximum gap: {flags['max_gap_hours']:.1f} hours")
    print(f"  Sufficient coverage: {flags['has_sufficient_coverage']}")
    print(f"  Acceptable gaps: {flags['has_acceptable_gaps']}")
    print(f"  Missing data imputed: {flags['missing_data_imputed']}")
    
    # Compare with dense data
    dense_data = create_sample_gps_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dense_flagged = harmonizer.apply_missing_data_flags(dense_data, 'gps')
    
    dense_flags = dense_flagged['missing_data_flags']
    print(f"\nDense data comparison:")
    print(f"  Coverage ratio: {dense_flags['coverage_ratio']:.3f}")
    print(f"  Maximum gap: {dense_flags['max_gap_hours']:.1f} hours")
    print(f"  Sufficient coverage: {dense_flags['has_sufficient_coverage']}")
    print()


def example_custom_configuration():
    """Example showing custom harmonization configuration."""
    print("=== Custom Configuration Example ===")
    
    # Create custom configuration
    custom_config = HarmonizationConfig(
        gps_resample_freq=10,  # 10-minute intervals instead of 5
        accelerometer_resample_freq=2,  # 2-minute intervals instead of 1
        min_data_coverage=0.8,  # Stricter coverage requirement
        max_gap_tolerance_hours=2.0,  # Stricter gap tolerance
        target_timezone="EST"
    )
    
    # Create harmonizer with custom config
    harmonizer = DataHarmonizer(custom_config)
    
    print("Custom configuration parameters:")
    print(f"  GPS resample frequency: {custom_config.gps_resample_freq} minutes")
    print(f"  Accelerometer resample frequency: {custom_config.accelerometer_resample_freq} minutes")
    print(f"  Minimum data coverage: {custom_config.min_data_coverage}")
    print(f"  Maximum gap tolerance: {custom_config.max_gap_tolerance_hours} hours")
    print(f"  Target timezone: {custom_config.target_timezone}")
    
    # Apply harmonization with custom config
    gps_data = create_sample_gps_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = harmonizer.harmonize_gps_data(gps_data)
    
    metadata = result['harmonization_metadata']
    print(f"\nApplied harmonization parameters:")
    print(f"  Resample frequency: {metadata['resample_freq_minutes']} minutes")
    print(f"  Aggregation method: {metadata['aggregation_method']}")
    print()


def example_device_metadata_handling():
    """Example showing device metadata integration."""
    print("=== Device Metadata Handling Example ===")
    
    harmonizer = DataHarmonizer()
    gps_data = create_sample_gps_data()
    
    # Different device metadata scenarios
    device_scenarios = [
        {
            'name': 'Android Device',
            'metadata': {'device_type': 'android', 'timezone': 'UTC', 'os_version': '13'}
        },
        {
            'name': 'iOS Device',
            'metadata': {'device_type': 'ios', 'timezone': 'EST', 'os_version': '16.3'}
        },
        {
            'name': 'Unknown Device',
            'metadata': None
        }
    ]
    
    for scenario in device_scenarios:
        print(f"\n{scenario['name']}:")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = harmonizer.harmonize_gps_data(gps_data, scenario['metadata'])
        
        metadata = result['harmonization_metadata']
        print(f"  Device type: {metadata['device_type']}")
        print(f"  Original timezone: {metadata['original_timezone']}")
        print(f"  Resample frequency: {metadata['resample_freq_minutes']} minutes")
    
    print()


def example_provenance_tracking():
    """Example showing provenance tracking capabilities."""
    print("=== Provenance Tracking Example ===")
    
    harmonizer = DataHarmonizer()
    
    # Perform multiple operations
    gps_data = create_sample_gps_data()
    accel_data = create_sample_accelerometer_data()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Multiple harmonization operations
        harmonizer.harmonize_gps_data(gps_data)
        harmonizer.harmonize_accelerometer_data(accel_data)
        harmonizer.apply_temporal_segmentation(gps_data, 'gps')
        harmonizer.apply_missing_data_flags(gps_data, 'gps')
    
    # Get provenance log
    provenance = harmonizer.get_provenance_log()
    
    print(f"Total operations performed: {len(provenance)}")
    print()
    
    for i, entry in enumerate(provenance, 1):
        print(f"Operation {i}: {entry['operation']}")
        print(f"  Timestamp: {entry['timestamp']}")
        print(f"  Duration: {entry['duration_seconds']:.3f} seconds")
        print(f"  Input records: {entry['input_records']}")
        if 'output_records' in entry:
            print(f"  Output records: {entry['output_records']}")
        print(f"  Parameters: {entry['parameters']}")
        print()
    
    # Clear provenance and verify
    harmonizer.clear_provenance_log()
    print(f"After clearing: {len(harmonizer.get_provenance_log())} operations")
    print()


def example_complete_workflow():
    """Example showing complete harmonization workflow."""
    print("=== Complete Harmonization Workflow Example ===")
    
    # Initialize harmonizer
    harmonizer = DataHarmonizer()
    
    # Create all data types
    gps_data = create_sample_gps_data()
    accel_data = create_sample_accelerometer_data()
    comm_data = create_sample_communication_data()
    screen_data = create_sample_screen_state_data()
    
    print("Starting complete harmonization workflow...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Step 1: Harmonize all data types
        print("1. Harmonizing sensor data...")
        harmonized_gps = harmonizer.harmonize_gps_data(gps_data)
        harmonized_accel = harmonizer.harmonize_accelerometer_data(accel_data)
        harmonized_comm = harmonizer.harmonize_communication_data(comm_data)
        harmonized_screen = harmonizer.harmonize_screen_state_data(screen_data)
        
        # Step 2: Apply temporal segmentation
        print("2. Applying temporal segmentation...")
        segmented_gps = harmonizer.apply_temporal_segmentation(harmonized_gps, 'gps')
        segmented_accel = harmonizer.apply_temporal_segmentation(harmonized_accel, 'accelerometer')
        segmented_comm = harmonizer.apply_temporal_segmentation(harmonized_comm, 'communication')
        segmented_screen = harmonizer.apply_temporal_segmentation(harmonized_screen, 'screen_state')
        
        # Step 3: Apply missing data analysis
        print("3. Analyzing missing data patterns...")
        final_gps = harmonizer.apply_missing_data_flags(segmented_gps, 'gps')
        final_accel = harmonizer.apply_missing_data_flags(segmented_accel, 'accelerometer')
        final_comm = harmonizer.apply_missing_data_flags(segmented_comm, 'communication')
        final_screen = harmonizer.apply_missing_data_flags(segmented_screen, 'screen_state')
    
    # Summary of final results
    print("\nFinal harmonized data summary:")
    datasets = [
        ('GPS', final_gps),
        ('Accelerometer', final_accel),
        ('Communication', final_comm),
        ('Screen State', final_screen)
    ]
    
    for name, data in datasets:
        flags = data['missing_data_flags']
        print(f"  {name}:")
        print(f"    Records: {len(data['timestamp'])}")
        print(f"    Coverage: {flags['coverage_ratio']:.3f}")
        print(f"    Max gap: {flags['max_gap_hours']:.1f}h")
        print(f"    Sufficient data: {flags['has_sufficient_coverage']}")
    
    # Provenance summary
    provenance = harmonizer.get_provenance_log()
    print(f"\nTotal operations: {len(provenance)}")
    print(f"Total processing time: {sum(entry['duration_seconds'] for entry in provenance):.3f} seconds")
    
    print()


if __name__ == "__main__":
    """Run all harmonization examples."""
    print("Psyconstruct Data Harmonization Examples")
    print("=" * 60)
    
    example_basic_harmonization()
    example_temporal_segmentation()
    example_missing_data_handling()
    example_custom_configuration()
    example_device_metadata_handling()
    example_provenance_tracking()
    example_complete_workflow()
    
    print("All harmonization examples completed successfully!")
