"""
Unit tests for provenance tracking module.

Tests operation tracking, feature extraction provenance,
construct aggregation provenance, reproducibility verification,
and provenance export/import functionality.
"""

import pytest
import tempfile
import json
from datetime import datetime
from pathlib import Path
from psyconstruct.utils.provenance import (
    ProvenanceTracker,
    OperationProvenance,
    FeatureExtractionProvenance,
    ConstructAggregationProvenance,
    get_provenance_tracker,
    track_operation
)


class TestOperationProvenance:
    """Test OperationProvenance dataclass."""
    
    def test_operation_provenance_creation(self):
        """Test creating an operation provenance record."""
        operation = OperationProvenance(
            operation_id="test_op_001",
            operation_type="test_operation",
            timestamp="2026-02-21T12:00:00",
            duration_seconds=1.5,
            input_parameters={"param1": "value1"},
            output_summary={"result": "success"},
            software_version="0.1.0"
        )
        
        assert operation.operation_id == "test_op_001"
        assert operation.operation_type == "test_operation"
        assert operation.duration_seconds == 1.5
        assert operation.parent_operations == []  # Default value
        assert operation.metadata == {}  # Default value
    
    def test_operation_provenance_with_optional_fields(self):
        """Test operation provenance with optional fields populated."""
        operation = OperationProvenance(
            operation_id="test_op_002",
            operation_type="test_operation",
            timestamp="2026-02-21T12:00:00",
            duration_seconds=2.0,
            input_parameters={},
            output_summary={},
            software_version="0.1.0",
            parent_operations=["parent_001"],
            metadata={"test": "metadata"}
        )
        
        assert operation.parent_operations == ["parent_001"]
        assert operation.metadata == {"test": "metadata"}


class TestFeatureExtractionProvenance:
    """Test FeatureExtractionProvenance dataclass."""
    
    def test_feature_extraction_provenance_creation(self):
        """Test creating a feature extraction provenance record."""
        provenance = FeatureExtractionProvenance(
            feature_name="activity_volume",
            construct="behavioral_activation",
            extraction_timestamp="2026-02-21T12:00:00",
            input_data_summary={"records": 1000},
            computation_parameters={"window_size": "1h"},
            result_summary={"mean_value": 45.2},
            data_quality_metrics={"completeness": 0.95},
            algorithm_version="1.0.1",
            computational_hash="abc123"
        )
        
        assert provenance.feature_name == "activity_volume"
        assert provenance.construct == "behavioral_activation"
        assert provenance.computational_hash == "abc123"
    
    def test_feature_extraction_validation(self):
        """Test validation of required fields."""
        # Missing feature name
        with pytest.raises(ValueError, match="Feature name is required"):
            FeatureExtractionProvenance(
                feature_name="",
                construct="test",
                extraction_timestamp="2026-02-21T12:00:00",
                input_data_summary={},
                computation_parameters={},
                result_summary={},
                data_quality_metrics={},
                algorithm_version="1.0.1",
                computational_hash="abc123"
            )
        
        # Missing construct name
        with pytest.raises(ValueError, match="Construct name is required"):
            FeatureExtractionProvenance(
                feature_name="test_feature",
                construct="",
                extraction_timestamp="2026-02-21T12:00:00",
                input_data_summary={},
                computation_parameters={},
                result_summary={},
                data_quality_metrics={},
                algorithm_version="1.0.1",
                computational_hash="abc123"
            )


class TestConstructAggregationProvenance:
    """Test ConstructAggregationProvenance dataclass."""
    
    def test_construct_aggregation_provenance_creation(self):
        """Test creating a construct aggregation provenance record."""
        provenance = ConstructAggregationProvenance(
            construct_name="behavioral_activation",
            aggregation_timestamp="2026-02-21T12:00:00",
            input_features=["activity_volume", "location_diversity"],
            aggregation_method="mean",
            feature_weights={"activity_volume": 0.5, "location_diversity": 0.5},
            normalization_applied=True,
            measurement_model="reflective",
            result_summary={"construct_score": 0.75},
            aggregation_hash="def456"
        )
        
        assert provenance.construct_name == "behavioral_activation"
        assert provenance.measurement_model == "reflective"
        assert provenance.aggregation_hash == "def456"
    
    def test_weight_validation(self):
        """Test validation of feature weights."""
        # Valid weights (sum to 1.0)
        valid_weights = {"feature1": 0.5, "feature2": 0.5}
        provenance = ConstructAggregationProvenance(
            construct_name="test",
            aggregation_timestamp="2026-02-21T12:00:00",
            input_features=["feature1", "feature2"],
            aggregation_method="mean",
            feature_weights=valid_weights,
            normalization_applied=False,
            measurement_model="reflective",
            result_summary={},
            aggregation_hash="test"
        )
        assert provenance.feature_weights == valid_weights
        
        # Invalid weights (sum != 1.0)
        invalid_weights = {"feature1": 0.6, "feature2": 0.5}  # Sum = 1.1
        with pytest.raises(ValueError, match="Feature weights must sum to 1.0"):
            ConstructAggregationProvenance(
                construct_name="test",
                aggregation_timestamp="2026-02-21T12:00:00",
                input_features=["feature1", "feature2"],
                aggregation_method="mean",
                feature_weights=invalid_weights,
                normalization_applied=False,
                measurement_model="reflective",
                result_summary={},
                aggregation_hash="test"
            )


class TestProvenanceTracker:
    """Test ProvenanceTracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization with default and custom session ID."""
        # Default initialization
        tracker = ProvenanceTracker()
        assert tracker.session_id is not None
        assert len(tracker.session_id) > 0
        assert len(tracker.operations) == 0
        assert len(tracker.feature_extractions) == 0
        assert len(tracker.construct_aggregations) == 0
        
        # Custom session ID
        custom_tracker = ProvenanceTracker(session_id="test_session_001")
        assert custom_tracker.session_id == "test_session_001"
    
    def test_start_and_complete_operation(self):
        """Test starting and completing operations."""
        tracker = ProvenanceTracker()
        
        # Start operation
        operation_id = tracker.start_operation(
            operation_type="test_operation",
            input_parameters={"param1": "value1"}
        )
        
        assert operation_id is not None
        assert len(tracker.operations) == 1
        
        # Get the operation
        operation = tracker.operations[0]
        assert operation.operation_type == "test_operation"
        assert operation.input_parameters == {"param1": "value1"}
        assert operation.duration_seconds == 0.0  # Not completed yet
        assert operation.output_summary == {}  # Not completed yet
        
        # Complete operation
        tracker.complete_operation(
            operation_id=operation_id,
            output_summary={"result": "success"},
            duration_seconds=1.5,
            random_seed=42
        )
        
        # Verify completion
        completed_operation = tracker.get_operation_provenance(operation_id)
        assert completed_operation.output_summary == {"result": "success"}
        assert completed_operation.duration_seconds == 1.5
        assert completed_operation.random_seed == 42
    
    def test_complete_nonexistent_operation(self):
        """Test completing an operation that doesn't exist."""
        tracker = ProvenanceTracker()
        
        with pytest.raises(ValueError, match="Operation ID not found"):
            tracker.complete_operation(
                operation_id="nonexistent_id",
                output_summary={},
                duration_seconds=1.0
            )
    
    def test_record_feature_extraction(self):
        """Test recording feature extraction provenance."""
        tracker = ProvenanceTracker()
        
        computational_hash = tracker.record_feature_extraction(
            feature_name="activity_volume",
            construct="behavioral_activation",
            input_data_summary={"records": 1000, "duration_hours": 24},
            computation_parameters={"window_size": "1h", "aggregation": "sum"},
            result_summary={"mean_value": 45.2, "std_value": 12.3},
            data_quality_metrics={"completeness": 0.95, "missing_rate": 0.05},
            algorithm_version="1.0.1"
        )
        
        assert computational_hash is not None
        assert len(computational_hash) == 64  # SHA256 hash length
        assert len(tracker.feature_extractions) == 1
        
        # Verify the record
        provenance = tracker.feature_extractions[0]
        assert provenance.feature_name == "activity_volume"
        assert provenance.construct == "behavioral_activation"
        assert provenance.computational_hash == computational_hash
        assert provenance.algorithm_version == "1.0.1"
    
    def test_record_construct_aggregation(self):
        """Test recording construct aggregation provenance."""
        tracker = ProvenanceTracker()
        
        aggregation_hash = tracker.record_construct_aggregation(
            construct_name="behavioral_activation",
            input_features=["activity_volume", "location_diversity", "app_usage_breadth", "activity_timing_variance"],
            aggregation_method="weighted_mean",
            feature_weights={"activity_volume": 0.25, "location_diversity": 0.25, "app_usage_breadth": 0.25, "activity_timing_variance": 0.25},
            normalization_applied=True,
            measurement_model="reflective",
            result_summary={"construct_score": 0.72, "confidence": 0.85}
        )
        
        assert aggregation_hash is not None
        assert len(aggregation_hash) == 64  # SHA256 hash length
        assert len(tracker.construct_aggregations) == 1
        
        # Verify the record
        provenance = tracker.construct_aggregations[0]
        assert provenance.construct_name == "behavioral_activation"
        assert provenance.measurement_model == "reflective"
        assert provenance.normalization_applied == True
        assert len(provenance.input_features) == 4
    
    def test_get_feature_provenance(self):
        """Test retrieving provenance for specific features."""
        tracker = ProvenanceTracker()
        
        # Record multiple feature extractions
        tracker.record_feature_extraction(
            feature_name="activity_volume",
            construct="behavioral_activation",
            input_data_summary={},
            computation_parameters={},
            result_summary={},
            data_quality_metrics={},
            algorithm_version="1.0.1"
        )
        
        tracker.record_feature_extraction(
            feature_name="location_diversity",
            construct="behavioral_activation",
            input_data_summary={},
            computation_parameters={},
            result_summary={},
            data_quality_metrics={},
            algorithm_version="1.0.1"
        )
        
        tracker.record_feature_extraction(
            feature_name="activity_volume",  # Same feature again
            construct="behavioral_activation",
            input_data_summary={},
            computation_parameters={},
            result_summary={},
            data_quality_metrics={},
            algorithm_version="1.0.1"
        )
        
        # Get provenance for specific feature
        activity_volume_provenance = tracker.get_feature_provenance("activity_volume")
        location_diversity_provenance = tracker.get_feature_provenance("location_diversity")
        
        assert len(activity_volume_provenance) == 2  # Two extractions
        assert len(location_diversity_provenance) == 1  # One extraction
        
        # Verify all returned records have the correct feature name
        for provenance in activity_volume_provenance:
            assert provenance.feature_name == "activity_volume"
        
        for provenance in location_diversity_provenance:
            assert provenance.feature_name == "location_diversity"
    
    def test_get_construct_provenance(self):
        """Test retrieving provenance for specific constructs."""
        tracker = ProvenanceTracker()
        
        # Record construct aggregations
        tracker.record_construct_aggregation(
            construct_name="behavioral_activation",
            input_features=["activity_volume"],
            aggregation_method="mean",
            feature_weights={"activity_volume": 1.0},
            normalization_applied=False,
            measurement_model="reflective",
            result_summary={}
        )
        
        tracker.record_construct_aggregation(
            construct_name="avoidance",
            input_features=["home_confinement"],
            aggregation_method="mean",
            feature_weights={"home_confinement": 1.0},
            normalization_applied=False,
            measurement_model="reflective",
            result_summary={}
        )
        
        # Get provenance for specific constructs
        ba_provenance = tracker.get_construct_provenance("behavioral_activation")
        av_provenance = tracker.get_construct_provenance("avoidance")
        
        assert len(ba_provenance) == 1
        assert len(av_provenance) == 1
        
        assert ba_provenance[0].construct_name == "behavioral_activation"
        assert av_provenance[0].construct_name == "avoidance"
    
    def test_export_and_import_provenance(self):
        """Test exporting and importing provenance records."""
        tracker = ProvenanceTracker(session_id="test_export_session")
        
        # Add some operations and records
        operation_id = tracker.start_operation("test_op", {})
        tracker.complete_operation(operation_id, {"result": "success"}, 1.0)
        
        tracker.record_feature_extraction(
            feature_name="test_feature",
            construct="test_construct",
            input_data_summary={},
            computation_parameters={},
            result_summary={},
            data_quality_metrics={},
            algorithm_version="1.0.1"
        )
        
        # Export provenance
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            exported_data = tracker.export_provenance(export_path)
            
            # Verify exported data structure
            assert 'session_metadata' in exported_data
            assert 'session_id' in exported_data
            assert 'operations' in exported_data
            assert 'feature_extractions' in exported_data
            assert 'construct_aggregations' in exported_data
            assert 'summary_statistics' in exported_data
            
            assert exported_data['session_id'] == "test_export_session"
            assert len(exported_data['operations']) == 1
            assert len(exported_data['feature_extractions']) == 1
            
            # Verify file was created
            assert export_path.exists()
            
            # Import provenance into new tracker
            new_tracker = ProvenanceTracker()
            new_tracker.import_provenance(export_path)
            
            # Verify imported data
            assert new_tracker.session_id == "test_export_session"
            assert len(new_tracker.operations) == 1
            assert len(new_tracker.feature_extractions) == 1
            
            # Verify specific content
            imported_operation = new_tracker.operations[0]
            assert imported_operation.operation_type == "test_op"
            assert imported_operation.output_summary == {"result": "success"}
            
        finally:
            if export_path.exists():
                export_path.unlink()
    
    def test_import_nonexistent_file(self):
        """Test importing from nonexistent file."""
        tracker = ProvenanceTracker()
        
        with pytest.raises(FileNotFoundError):
            tracker.import_provenance(Path("nonexistent_file.json"))
    
    def test_import_invalid_json(self):
        """Test importing invalid JSON file."""
        tracker = ProvenanceTracker()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json content}')
            invalid_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Invalid provenance file format"):
                tracker.import_provenance(invalid_path)
        finally:
            invalid_path.unlink()
    
    def test_clear_provenance(self):
        """Test clearing provenance records."""
        tracker = ProvenanceTracker()
        
        # Add some records
        tracker.start_operation("test_op", {})
        tracker.record_feature_extraction(
            feature_name="test_feature",
            construct="test_construct",
            input_data_summary={},
            computation_parameters={},
            result_summary={},
            data_quality_metrics={},
            algorithm_version="1.0.1"
        )
        
        assert len(tracker.operations) == 1
        assert len(tracker.feature_extractions) == 1
        
        # Clear provenance
        tracker.clear_provenance()
        
        assert len(tracker.operations) == 0
        assert len(tracker.feature_extractions) == 0
        assert len(tracker.construct_aggregations) == 0
    
    def test_get_session_summary(self):
        """Test getting session summary."""
        tracker = ProvenanceTracker()
        
        # Add some records
        op_id1 = tracker.start_operation("op1", {})
        tracker.complete_operation(op_id1, {}, 1.0)
        
        op_id2 = tracker.start_operation("op2", {})
        tracker.complete_operation(op_id2, {}, 2.0)
        
        tracker.record_feature_extraction(
            feature_name="feature1",
            construct="construct1",
            input_data_summary={},
            computation_parameters={},
            result_summary={},
            data_quality_metrics={},
            algorithm_version="1.0.1"
        )
        
        tracker.record_feature_extraction(
            feature_name="feature2",
            construct="construct1",
            input_data_summary={},
            computation_parameters={},
            result_summary={},
            data_quality_metrics={},
            algorithm_version="1.0.1"
        )
        
        tracker.record_construct_aggregation(
            construct_name="construct1",
            input_features=["feature1", "feature2"],
            aggregation_method="mean",
            feature_weights={"feature1": 0.5, "feature2": 0.5},
            normalization_applied=False,
            measurement_model="reflective",
            result_summary={}
        )
        
        # Get summary
        summary = tracker.get_session_summary()
        
        assert summary['total_operations'] == 2
        assert summary['total_feature_extractions'] == 2
        assert summary['total_construct_aggregations'] == 1
        assert summary['unique_features_extracted'] == 2
        assert summary['unique_constructs_processed'] == 1
        assert 'session_duration_seconds' in summary
    
    def test_computational_hash_consistency(self):
        """Test that computational hashes are consistent for identical inputs."""
        tracker = ProvenanceTracker()
        
        # Record same feature extraction twice
        hash1 = tracker.record_feature_extraction(
            feature_name="test_feature",
            construct="test_construct",
            input_data_summary={"records": 1000},
            computation_parameters={"window": "1h"},
            result_summary={"mean": 45.0},
            data_quality_metrics={"completeness": 0.95},
            algorithm_version="1.0.1"
        )
        
        hash2 = tracker.record_feature_extraction(
            feature_name="test_feature",
            construct="test_construct",
            input_data_summary={"records": 1000},
            computation_parameters={"window": "1h"},
            result_summary={"mean": 45.0},
            data_quality_metrics={"completeness": 0.95},
            algorithm_version="1.0.1"
        )
        
        # Hashes should be identical for identical inputs
        assert hash1 == hash2
        
        # Different parameters should produce different hash
        hash3 = tracker.record_feature_extraction(
            feature_name="test_feature",
            construct="test_construct",
            input_data_summary={"records": 1000},
            computation_parameters={"window": "2h"},  # Different window
            result_summary={"mean": 45.0},
            data_quality_metrics={"completeness": 0.95},
            algorithm_version="1.0.1"
        )
        
        assert hash3 != hash1


class TestGlobalTracker:
    """Test global provenance tracker functionality."""
    
    def test_get_global_tracker(self):
        """Test getting global tracker instance."""
        # Clear global tracker
        import psyconstruct.utils.provenance
        psyconstruct.utils.provenance._global_tracker = None
        
        # Get tracker (should create new instance)
        tracker1 = get_provenance_tracker()
        assert isinstance(tracker1, ProvenanceTracker)
        
        # Get tracker again (should return same instance)
        tracker2 = get_provenance_tracker()
        assert tracker1 is tracker2
        
        # Get tracker with custom session ID (should create new instance)
        tracker3 = get_provenance_tracker(session_id="custom_session")
        assert tracker3.session_id == "custom_session"
        assert tracker3 is not tracker1


class TestProvenanceIntegration:
    """Integration tests for provenance tracking."""
    
    def test_complete_workflow_provenance(self):
        """Test provenance tracking for complete workflow."""
        tracker = ProvenanceTracker()
        
        # Step 1: Data harmonization operation
        harmonize_op_id = tracker.start_operation(
            operation_type="harmonize_gps_data",
            input_parameters={"resample_freq": 5, "aggregation": "median"}
        )
        tracker.complete_operation(
            operation_id=harmonize_op_id,
            output_summary={"output_records": 100, "success": True},
            duration_seconds=2.5
        )
        
        # Step 2: Feature extraction
        feature_hash = tracker.record_feature_extraction(
            feature_name="location_diversity",
            construct="behavioral_activation",
            input_data_summary={"gps_records": 100, "time_span_hours": 168},
            computation_parameters={"clustering_radius": 50, "min_points": 5},
            result_summary={"entropy_value": 2.34, "unique_locations": 15},
            data_quality_metrics={"completeness": 0.92, "coverage_ratio": 0.88},
            algorithm_version="1.0.1"
        )
        
        # Step 3: Another feature extraction
        activity_hash = tracker.record_feature_extraction(
            feature_name="activity_volume",
            construct="behavioral_activation",
            input_data_summary={"accel_records": 1000, "time_span_hours": 24},
            computation_parameters={"window_size": "1h", "aggregation": "sum"},
            result_summary={"mean_activity": 45.2, "peak_activity": 89.1},
            data_quality_metrics={"completeness": 0.95, "sampling_rate": 1.0},
            algorithm_version="1.0.1"
        )
        
        # Step 4: Construct aggregation
        construct_hash = tracker.record_construct_aggregation(
            construct_name="behavioral_activation",
            input_features=["location_diversity", "activity_volume"],
            aggregation_method="weighted_mean",
            feature_weights={"location_diversity": 0.5, "activity_volume": 0.5},
            normalization_applied=True,
            measurement_model="reflective",
            result_summary={"construct_score": 0.67, "confidence_interval": [0.61, 0.73]}
        )
        
        # Verify complete provenance
        assert len(tracker.operations) == 1
        assert len(tracker.feature_extractions) == 2
        assert len(tracker.construct_aggregations) == 1
        
        # Export and verify structure
        exported = tracker.export_provenance()
        assert exported['summary_statistics']['operation_statistics']['total_operations'] == 1
        assert exported['summary_statistics']['feature_statistics']['total_extractions'] == 2
        assert exported['summary_statistics']['construct_statistics']['total_aggregations'] == 1
        
        # Verify session summary
        summary = tracker.get_session_summary()
        assert summary['total_operations'] == 1
        assert summary['total_feature_extractions'] == 2
        assert summary['total_construct_aggregations'] == 1
        assert summary['unique_features_extracted'] == 2
        assert summary['unique_constructs_processed'] == 1
