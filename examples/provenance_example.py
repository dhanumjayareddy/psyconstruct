"""
Example usage of provenance tracking module.

This example demonstrates how to:
1. Track operations with automatic provenance
2. Record feature extraction provenance
3. Record construct aggregation provenance
4. Verify reproducibility
5. Export and import provenance records
"""

import sys
from pathlib import Path
import time
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psyconstruct.utils.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
    track_operation
)


def example_basic_operation_tracking():
    """Example showing basic operation tracking."""
    print("=== Basic Operation Tracking Example ===")
    
    tracker = ProvenanceTracker(session_id="basic_tracking_example")
    
    # Simulate data harmonization operation
    print("Starting data harmonization operation...")
    harmonize_id = tracker.start_operation(
        operation_type="harmonize_gps_data",
        input_parameters={
            "resample_freq_minutes": 5,
            "aggregation_method": "median",
            "device_type": "android"
        }
    )
    
    # Simulate operation execution
    time.sleep(0.1)  # Simulate processing time
    
    # Complete operation
    tracker.complete_operation(
        operation_id=harmonize_id,
        output_summary={
            "input_records": 1000,
            "output_records": 200,
            "success": True,
            "data_quality_score": 0.95
        },
        duration_seconds=0.1,
        random_seed=42
    )
    
    # Get operation provenance
    operation = tracker.get_operation_provenance(harmonize_id)
    print(f"Operation completed:")
    print(f"  Type: {operation.operation_type}")
    print(f"  Duration: {operation.duration_seconds:.3f} seconds")
    print(f"  Input records: {operation.input_parameters}")
    print(f"  Output summary: {operation.output_summary}")
    print(f"  Random seed: {operation.random_seed}")
    print()


def example_feature_extraction_provenance():
    """Example showing feature extraction provenance tracking."""
    print("=== Feature Extraction Provenance Example ===")
    
    tracker = ProvenanceTracker()
    
    # Record activity volume feature extraction
    print("Recording activity volume feature extraction...")
    activity_hash = tracker.record_feature_extraction(
        feature_name="activity_volume",
        construct="behavioral_activation",
        input_data_summary={
            "accelerometer_records": 1440,
            "time_span_hours": 24,
            "sampling_rate_hz": 1.0,
            "missing_data_rate": 0.02
        },
        computation_parameters={
            "window_size": "1h",
            "aggregation_method": "sum",
            "magnitude_computed": True
        },
        result_summary={
            "daily_mean": 45.2,
            "daily_std": 12.3,
            "peak_value": 89.1,
            "min_value": 8.7
        },
        data_quality_metrics={
            "completeness": 0.98,
            "coverage_ratio": 0.96,
            "outlier_rate": 0.01
        },
        algorithm_version="1.0.0"
    )
    
    # Record location diversity feature extraction
    print("Recording location diversity feature extraction...")
    location_hash = tracker.record_feature_extraction(
        feature_name="location_diversity",
        construct="behavioral_activation",
        input_data_summary={
            "gps_records": 500,
            "time_span_hours": 168,  # 1 week
            "unique_locations_raw": 25
        },
        computation_parameters={
            "clustering_radius_meters": 50,
            "min_cluster_size": 5,
            "entropy_calculation": "shannon"
        },
        result_summary={
            "weekly_entropy": 2.34,
            "unique_locations_clustered": 8,
            "location_variety_index": 0.67
        },
        data_quality_metrics={
            "completeness": 0.85,
            "coverage_ratio": 0.78,
            "gps_accuracy_mean": 5.2
        },
        algorithm_version="1.0.0"
    )
    
    print(f"Activity volume hash: {activity_hash[:16]}...")
    print(f"Location diversity hash: {location_hash[:16]}...")
    
    # Get provenance for specific features
    activity_provenance = tracker.get_feature_provenance("activity_volume")
    location_provenance = tracker.get_feature_provenance("location_diversity")
    
    print(f"\nActivity volume extractions: {len(activity_provenance)}")
    print(f"Location diversity extractions: {len(location_provenance)}")
    
    # Show detailed provenance
    if activity_provenance:
        prov = activity_provenance[0]
        print(f"\nActivity volume provenance:")
        print(f"  Extraction timestamp: {prov.extraction_timestamp}")
        print(f"  Input records: {prov.input_data_summary['accelerometer_records']}")
        print(f"  Result mean: {prov.result_summary['daily_mean']}")
        print(f"  Data completeness: {prov.data_quality_metrics['completeness']}")
    
    print()


def example_construct_aggregation_provenance():
    """Example showing construct aggregation provenance tracking."""
    print("=== Construct Aggregation Provenance Example ===")
    
    tracker = ProvenanceTracker()
    
    # Record behavioral activation aggregation
    print("Recording behavioral activation construct aggregation...")
    ba_hash = tracker.record_construct_aggregation(
        construct_name="behavioral_activation",
        input_features=["activity_volume", "location_diversity", "app_usage_breadth", "activity_timing_variance"],
        aggregation_method="weighted_mean",
        feature_weights={
            "activity_volume": 0.25,
            "location_diversity": 0.25,
            "app_usage_breadth": 0.25,
            "activity_timing_variance": 0.25
        },
        normalization_applied=True,
        measurement_model="reflective",
        result_summary={
            "construct_score": 0.72,
            "confidence_interval": [0.68, 0.76],
            "feature_contributions": {
                "activity_volume": 0.18,
                "location_diversity": 0.19,
                "app_usage_breadth": 0.17,
                "activity_timing_variance": 0.18
            }
        }
    )
    
    # Record social engagement aggregation (formative model)
    print("Recording social engagement construct aggregation...")
    se_hash = tracker.record_construct_aggregation(
        construct_name="social_engagement",
        input_features=["communication_frequency", "contact_diversity", "initiation_rate"],
        aggregation_method="weighted_mean",
        feature_weights={
            "communication_frequency": 0.33,
            "contact_diversity": 0.33,
            "initiation_rate": 0.34
        },
        normalization_applied=True,
        measurement_model="formative",
        result_summary={
            "construct_score": 0.58,
            "confidence_interval": [0.52, 0.64],
            "feature_contributions": {
                "communication_frequency": 0.19,
                "contact_diversity": 0.19,
                "initiation_rate": 0.20
            }
        }
    )
    
    print(f"Behavioral activation hash: {ba_hash[:16]}...")
    print(f"Social engagement hash: {se_hash[:16]}...")
    
    # Get construct provenance
    ba_provenance = tracker.get_construct_provenance("behavioral_activation")
    se_provenance = tracker.get_construct_provenance("social_engagement")
    
    print(f"\nBehavioral activation aggregations: {len(ba_provenance)}")
    print(f"Social engagement aggregations: {len(se_provenance)}")
    
    # Show detailed provenance
    if ba_provenance:
        prov = ba_provenance[0]
        print(f"\nBehavioral activation provenance:")
        print(f"  Aggregation timestamp: {prov.aggregation_timestamp}")
        print(f"  Measurement model: {prov.measurement_model}")
        print(f"  Normalization applied: {prov.normalization_applied}")
        print(f"  Construct score: {prov.result_summary['construct_score']}")
        print(f"  Input features: {', '.join(prov.input_features)}")
    
    print()


def example_reproducibility_verification():
    """Example showing reproducibility verification."""
    print("=== Reproducibility Verification Example ===")
    
    tracker = ProvenanceTracker()
    
    # Original feature extraction
    print("Performing original feature extraction...")
    original_hash = tracker.record_feature_extraction(
        feature_name="activity_volume",
        construct="behavioral_activation",
        input_data_summary={"records": 1000},
        computation_parameters={
            "window_size": "1h",
            "aggregation": "sum",
            "normalization": "z_score"
        },
        result_summary={"mean": 45.2},
        data_quality_metrics={"completeness": 0.95},
        algorithm_version="1.0.0"
    )
    
    # Test reproducibility with identical parameters
    print("Testing reproducibility with identical parameters...")
    is_reproducible_same = tracker.verify_reproducibility(
        feature_name="activity_volume",
        construct="behavioral_activation",
        computation_parameters={
            "window_size": "1h",
            "aggregation": "sum",
            "normalization": "z_score"
        },
        expected_hash=original_hash
    )
    
    # Test reproducibility with different parameters
    print("Testing reproducibility with different parameters...")
    is_reproducible_diff = tracker.verify_reproducibility(
        feature_name="activity_volume",
        construct="behavioral_activation",
        computation_parameters={
            "window_size": "2h",  # Different window size
            "aggregation": "sum",
            "normalization": "z_score"
        },
        expected_hash=original_hash
    )
    
    print(f"Original hash: {original_hash[:16]}...")
    print(f"Reproducible with same parameters: {is_reproducible_same}")
    print(f"Reproducible with different parameters: {is_reproducible_diff}")
    
    # Multiple extractions with same parameters
    print("\nPerforming multiple extractions with same parameters...")
    hash2 = tracker.record_feature_extraction(
        feature_name="activity_volume",
        construct="behavioral_activation",
        input_data_summary={"records": 1000},
        computation_parameters={
            "window_size": "1h",
            "aggregation": "sum",
            "normalization": "z_score"
        },
        result_summary={"mean": 45.2},
        data_quality_metrics={"completeness": 0.95},
        algorithm_version="1.0.0"
    )
    
    print(f"Hash 1: {original_hash[:16]}...")
    print(f"Hash 2: {hash2[:16]}...")
    print(f"Hashes identical: {original_hash == hash2}")
    
    print()


def example_provenance_export_import():
    """Example showing provenance export and import."""
    print("=== Provenance Export/Import Example ===")
    
    # Create tracker with some data
    original_tracker = ProvenanceTracker(session_id="export_test_session")
    
    # Add some operations and records
    op_id = original_tracker.start_operation("test_operation", {"param": "value"})
    original_tracker.complete_operation(op_id, {"result": "success"}, 1.5)
    
    original_tracker.record_feature_extraction(
        feature_name="test_feature",
        construct="test_construct",
        input_data_summary={},
        computation_parameters={},
        result_summary={},
        data_quality_metrics={},
        algorithm_version="1.0.0"
    )
    
    print(f"Original tracker session ID: {original_tracker.session_id}")
    print(f"Original operations: {len(original_tracker.operations)}")
    print(f"Original feature extractions: {len(original_tracker.feature_extractions)}")
    
    # Export provenance to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        export_path = Path(f.name)
    
    try:
        print(f"\nExporting provenance to: {export_path}")
        exported_data = original_tracker.export_provenance(export_path)
        
        print(f"Export completed successfully")
        print(f"Exported session ID: {exported_data['session_id']}")
        print(f"Exported operations: {len(exported_data['operations'])}")
        print(f"Exported feature extractions: {len(exported_data['feature_extractions'])}")
        
        # Import provenance into new tracker
        print(f"\nImporting provenance into new tracker...")
        new_tracker = ProvenanceTracker()
        new_tracker.import_provenance(export_path)
        
        print(f"Import completed successfully")
        print(f"Imported session ID: {new_tracker.session_id}")
        print(f"Imported operations: {len(new_tracker.operations)}")
        print(f"Imported feature extractions: {len(new_tracker.feature_extractions)}")
        
        # Verify data integrity
        assert original_tracker.session_id == new_tracker.session_id
        assert len(original_tracker.operations) == len(new_tracker.operations)
        assert len(original_tracker.feature_extractions) == len(new_tracker.feature_extractions)
        
        print(f"\nData integrity verified: ✓")
        
    finally:
        # Clean up temporary file
        if export_path.exists():
            export_path.unlink()
    
    print()


def example_decorator_tracking():
    """Example showing automatic operation tracking with decorator."""
    print("=== Decorator Tracking Example ===")
    
    # Clear global tracker
    import psyconstruct.utils.provenance
    psyconstruct.utils.provenance._global_tracker = None
    
    @track_operation("data_preprocessing", {"stage": "harmonization"})
    def preprocess_data(data_size, quality_threshold):
        """Simulate data preprocessing function."""
        print(f"  Preprocessing {data_size} records...")
        time.sleep(0.05)  # Simulate processing
        if data_size < quality_threshold:
            raise ValueError("Data size below quality threshold")
        return {"processed_records": data_size, "quality_score": 0.95}
    
    @track_operation("feature_computation", {"feature": "activity_volume"})
    def compute_feature(data, window_size):
        """Simulate feature computation function."""
        print(f"  Computing feature with window size {window_size}...")
        time.sleep(0.03)  # Simulate processing
        return {"feature_value": 45.2, "confidence": 0.88}
    
    try:
        # Successful operations
        print("Running successful operations...")
        result1 = preprocess_data(1000, 500)
        result2 = compute_feature(result1, "1h")
        
        print(f"Preprocessing result: {result1}")
        print(f"Feature computation result: {result2}")
        
        # Failed operation
        print("\nRunning failing operation...")
        preprocess_data(100, 500)  # This will fail
        
    except ValueError as e:
        print(f"Expected error caught: {e}")
    
    # Check provenance
    tracker = get_provenance_tracker()
    print(f"\nTracked operations: {len(tracker.operations)}")
    
    for i, op in enumerate(tracker.operations, 1):
        print(f"  Operation {i}: {op.operation_type}")
        print(f"    Success: {op.output_summary['success']}")
        print(f"    Duration: {op.duration_seconds:.3f}s")
        if not op.output_summary['success']:
            print(f"    Error: {op.output_summary['error_message']}")
    
    print()


def example_session_summary():
    """Example showing session summary and statistics."""
    print("=== Session Summary Example ===")
    
    tracker = ProvenanceTracker(session_id="summary_example")
    
    # Add various operations and records
    operations = [
        ("harmonize_gps_data", {"freq": 5}),
        ("harmonize_accelerometer", {"freq": 1}),
        ("extract_activity_volume", {"window": "1h"}),
        ("extract_location_diversity", {"radius": 50}),
        ("aggregate_behavioral_activation", {"model": "reflective"})
    ]
    
    for op_type, params in operations:
        op_id = tracker.start_operation(op_type, params)
        time.sleep(0.01)  # Simulate processing
        tracker.complete_operation(op_id, {"success": True}, 0.01)
    
    # Add feature extractions
    features = [
        ("activity_volume", "behavioral_activation"),
        ("location_diversity", "behavioral_activation"),
        ("communication_frequency", "social_engagement"),
        ("sleep_onset_consistency", "routine_stability")
    ]
    
    for feature, construct in features:
        tracker.record_feature_extraction(
            feature_name=feature,
            construct=construct,
            input_data_summary={"records": 1000},
            computation_parameters={"version": "1.0"},
            result_summary={"value": 0.5},
            data_quality_metrics={"completeness": 0.9},
            algorithm_version="1.0.0"
        )
    
    # Add construct aggregations
    constructs = ["behavioral_activation", "social_engagement", "routine_stability"]
    for construct in constructs:
        tracker.record_construct_aggregation(
            construct_name=construct,
            input_features=["feature1", "feature2"],
            aggregation_method="weighted_mean",
            feature_weights={"feature1": 0.5, "feature2": 0.5},
            normalization_applied=True,
            measurement_model="reflective",
            result_summary={"score": 0.7}
        )
    
    # Get session summary
    summary = tracker.get_session_summary()
    
    print("Session Summary:")
    print(f"  Session ID: {summary['session_id']}")
    print(f"  Session duration: {summary['session_duration_seconds']:.2f} seconds")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Total feature extractions: {summary['total_feature_extractions']}")
    print(f"  Total construct aggregations: {summary['total_construct_aggregations']}")
    print(f"  Unique features extracted: {summary['unique_features_extracted']}")
    print(f"  Unique constructs processed: {summary['unique_constructs_processed']}")
    
    # Export with summary statistics
    exported = tracker.export_provenance()
    stats = exported['summary_statistics']
    
    print("\nDetailed Statistics:")
    print(f"  Operation Statistics:")
    print(f"    Total operations: {stats['operation_statistics']['total_operations']}")
    print(f"    Total duration: {stats['operation_statistics']['total_duration_seconds']:.3f}s")
    print(f"    Average duration: {stats['operation_statistics']['average_duration_seconds']:.3f}s")
    
    print(f"  Feature Statistics:")
    print(f"    Total extractions: {stats['feature_statistics']['total_extractions']}")
    print(f"    Unique features: {stats['feature_statistics']['unique_features']}")
    
    print(f"  Construct Statistics:")
    print(f"    Total aggregations: {stats['construct_statistics']['total_aggregations']}")
    print(f"    Unique constructs: {stats['construct_statistics']['unique_constructs']}")
    print(f"    Measurement models: {', '.join(stats['construct_statistics']['measurement_models'])}")
    
    print()


def example_complete_workflow():
    """Example showing complete provenance tracking workflow."""
    print("=== Complete Provenance Workflow Example ===")
    
    tracker = ProvenanceTracker(session_id="complete_workflow_demo")
    
    print("Starting complete digital phenotyping workflow...")
    
    # Step 1: Data harmonization
    print("\n1. Data Harmonization Phase")
    harmonize_ops = ["harmonize_gps", "harmonize_accelerometer", "harmonize_communication", "harmonize_screen"]
    
    for op in harmonize_ops:
        op_id = tracker.start_operation(op, {"quality_check": True})
        time.sleep(0.02)
        tracker.complete_operation(op_id, {"records_processed": 1000, "success": True}, 0.02)
    
    # Step 2: Feature extraction
    print("2. Feature Extraction Phase")
    feature_configs = [
        ("activity_volume", "behavioral_activation", {"window": "1h"}),
        ("location_diversity", "behavioral_activation", {"radius": 50}),
        ("communication_frequency", "social_engagement", {"period": "daily"}),
        ("sleep_onset_consistency", "routine_stability", {"method": "screen_off"})
    ]
    
    for feature, construct, params in feature_configs:
        tracker.record_feature_extraction(
            feature_name=feature,
            construct=construct,
            input_data_summary={"records": 1000, "quality": 0.95},
            computation_parameters=params,
            result_summary={"value": 0.65, "ci": [0.60, 0.70]},
            data_quality_metrics={"completeness": 0.92, "accuracy": 0.88},
            algorithm_version="1.0.0"
        )
    
    # Step 3: Construct aggregation
    print("3. Construct Aggregation Phase")
    construct_configs = [
        ("behavioral_activation", "reflective", ["activity_volume", "location_diversity"]),
        ("social_engagement", "formative", ["communication_frequency"]),
        ("routine_stability", "reflective", ["sleep_onset_consistency"])
    ]
    
    for construct, model, features in construct_configs:
        weights = {f: 1.0/len(features) for f in features}
        tracker.record_construct_aggregation(
            construct_name=construct,
            input_features=features,
            aggregation_method="weighted_mean",
            feature_weights=weights,
            normalization_applied=True,
            measurement_model=model,
            result_summary={"construct_score": 0.68, "reliability": 0.85}
        )
    
    # Step 4: Quality verification
    print("4. Quality Verification Phase")
    
    # Verify reproducibility for one feature
    activity_provenance = tracker.get_feature_provenance("activity_volume")
    if activity_provenance:
        prov = activity_provenance[0]
        is_reproducible = tracker.verify_reproducibility(
            feature_name="activity_volume",
            construct="behavioral_activation",
            computation_parameters=prov.computation_parameters,
            expected_hash=prov.computational_hash
        )
        print(f"  Activity volume reproducibility: {is_reproducible}")
    
    # Generate final report
    print("\n5. Final Report")
    summary = tracker.get_session_summary()
    exported = tracker.export_provenance()
    
    print(f"  Session completed successfully")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Features extracted: {summary['total_feature_extractions']}")
    print(f"  Constructs aggregated: {summary['total_construct_aggregations']}")
    print(f"  Processing time: {summary['session_duration_seconds']:.2f} seconds")
    
    # Export final provenance
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        final_path = Path(f.name)
    
    try:
        tracker.export_provenance(final_path)
        print(f"  Provenance exported to: {final_path}")
        print(f"  File size: {final_path.stat().st_size} bytes")
    finally:
        if final_path.exists():
            final_path.unlink()
    
    print()


if __name__ == "__main__":
    """Run all provenance examples."""
    print("Psyconstruct Provenance Tracking Examples")
    print("=" * 60)
    
    example_basic_operation_tracking()
    example_feature_extraction_provenance()
    example_construct_aggregation_provenance()
    example_reproducibility_verification()
    example_provenance_export_import()
    example_decorator_tracking()
    example_session_summary()
    example_complete_workflow()
    
    print("All provenance examples completed successfully!")
