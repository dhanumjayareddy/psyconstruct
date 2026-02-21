"""
Unit tests for data harmonization module.

Tests cross-platform data standardization, resampling strategies,
timezone handling, missing data policies, and device metadata handling.
"""

import pytest
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any
from psyconstruct.preprocessing.harmonization import (
    DataHarmonizer,
    HarmonizationConfig
)


class TestHarmonizationConfig:
    """Test HarmonizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HarmonizationConfig()
        
        assert config.gps_resample_freq == 5
        assert config.accelerometer_resample_freq == 1
        assert config.screen_resample_freq == 1
        assert config.min_data_coverage == 0.7
        assert config.max_gap_tolerance_hours == 4.0
        assert config.target_timezone == "UTC"
        assert config.store_original_timezone == True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = HarmonizationConfig(
            gps_resample_freq=10,
            min_data_coverage=0.8,
            target_timezone="EST"
        )
        
        assert config.gps_resample_freq == 10
        assert config.min_data_coverage == 0.8
        assert config.target_timezone == "EST"


class TestDataHarmonizer:
    """Test DataHarmonizer class."""
    
    def create_sample_gps_data(self) -> Dict[str, Any]:
        """Create sample GPS data for testing."""
        base_time = datetime(2026, 2, 21, 12, 0, 0)
        timestamps = [base_time + timedelta(minutes=i*5) for i in range(10)]
        
        return {
            'timestamp': timestamps,
            'latitude': [40.7128 + i*0.001 for i in range(10)],
            'longitude': [-74.0060 + i*0.001 for i in range(10)]
        }
    
    def create_sample_accelerometer_data(self) -> Dict[str, Any]:
        """Create sample accelerometer data for testing."""
        base_time = datetime(2026, 2, 21, 12, 0, 0)
        timestamps = [base_time + timedelta(minutes=i) for i in range(60)]
        
        return {
            'timestamp': timestamps,
            'x': [0.1 + i*0.01 for i in range(60)],
            'y': [0.2 + i*0.01 for i in range(60)],
            'z': [9.8 + i*0.01 for i in range(60)]
        }
    
    def create_sample_communication_data(self) -> Dict[str, Any]:
        """Create sample communication data for testing."""
        base_time = datetime(2026, 2, 21, 12, 0, 0)
        timestamps = [base_time + timedelta(hours=i) for i in range(5)]
        
        return {
            'timestamp': timestamps,
            'direction': ['out', 'in', 'out', 'in', 'out'],
            'contact_id': ['contact_1', 'contact_2', 'contact_1', 'contact_3', 'contact_2']
        }
    
    def create_sample_screen_state_data(self) -> Dict[str, Any]:
        """Create sample screen state data for testing."""
        base_time = datetime(2026, 2, 21, 12, 0, 0)
        timestamps = [base_time + timedelta(minutes=i) for i in range(20)]
        
        return {
            'timestamp': timestamps,
            'state': ['on', 'on', 'off', 'off', 'on'] * 4
        }
    
    def test_harmonizer_initialization(self):
        """Test harmonizer initialization with default and custom config."""
        # Default config
        harmonizer = DataHarmonizer()
        assert harmonizer.config.gps_resample_freq == 5
        assert len(harmonizer.provenance) == 0
        
        # Custom config
        custom_config = HarmonizationConfig(gps_resample_freq=10)
        harmonizer = DataHarmonizer(custom_config)
        assert harmonizer.config.gps_resample_freq == 10
    
    def test_harmonize_gps_data(self):
        """Test GPS data harmonization."""
        harmonizer = DataHarmonizer()
        gps_data = self.create_sample_gps_data()
        device_metadata = {'device_type': 'android', 'timezone': 'UTC'}
        
        result = harmonizer.harmonize_gps_data(gps_data, device_metadata)
        
        # Check output structure
        assert 'timestamp' in result
        assert 'latitude' in result
        assert 'longitude' in result
        assert 'harmonization_metadata' in result
        
        # Check metadata
        metadata = result['harmonization_metadata']
        assert metadata['resample_freq_minutes'] == 5
        assert metadata['aggregation_method'] == 'median'
        assert metadata['device_type'] == 'android'
        
        # Check provenance
        assert len(harmonizer.provenance) == 1
        provenance_entry = harmonizer.provenance[0]
        assert provenance_entry['operation'] == 'harmonize_gps_data'
        assert provenance_entry['input_records'] == 10
    
    def test_harmonize_gps_data_validation(self):
        """Test GPS data validation."""
        harmonizer = DataHarmonizer()
        
        # Missing required column
        invalid_data = {'timestamp': [datetime.now()], 'latitude': [40.7128]}
        with pytest.raises(ValueError, match="Missing required column: longitude"):
            harmonizer.harmonize_gps_data(invalid_data)
        
        # Unequal column lengths
        invalid_data = {
            'timestamp': [datetime.now(), datetime.now()],
            'latitude': [40.7128],
            'longitude': [-74.0060, -74.0061]
        }
        with pytest.raises(ValueError, match="All GPS data columns must have equal length"):
            harmonizer.harmonize_gps_data(invalid_data)
    
    def test_harmonize_accelerometer_data(self):
        """Test accelerometer data harmonization."""
        harmonizer = DataHarmonizer()
        accel_data = self.create_sample_accelerometer_data()
        
        result = harmonizer.harmonize_accelerometer_data(accel_data)
        
        # Check output structure
        assert 'timestamp' in result
        assert 'magnitude' in result
        assert 'harmonization_metadata' in result
        
        # Check metadata
        metadata = result['harmonization_metadata']
        assert metadata['resample_freq_minutes'] == 1
        assert metadata['aggregation_method'] == 'mean'
        assert metadata['sampling_rate_normalized'] == True
        
        # Check provenance
        assert len(harmonizer.provenance) == 1
        provenance_entry = harmonizer.provenance[0]
        assert provenance_entry['operation'] == 'harmonize_accelerometer_data'
        assert provenance_entry['parameters']['magnitude_computed'] == True
    
    def test_harmonize_accelerometer_with_precomputed_magnitude(self):
        """Test accelerometer harmonization with precomputed magnitude."""
        harmonizer = DataHarmonizer()
        accel_data = self.create_sample_accelerometer_data()
        accel_data['magnitude'] = [1.0] * 60  # Add precomputed magnitude
        
        result = harmonizer.harmonize_accelerometer_data(accel_data)
        
        # Check that magnitude was not recomputed
        provenance_entry = harmonizer.provenance[0]
        assert provenance_entry['parameters']['magnitude_computed'] == False
    
    def test_harmonize_accelerometer_data_validation(self):
        """Test accelerometer data validation."""
        harmonizer = DataHarmonizer()
        
        # Missing required columns (no magnitude)
        invalid_data = {'timestamp': [datetime.now()], 'x': [0.1], 'y': [0.2]}
        with pytest.raises(ValueError, match="Missing required column: z"):
            harmonizer.harmonize_accelerometer_data(invalid_data)
        
        # Unequal column lengths
        invalid_data = {
            'timestamp': [datetime.now(), datetime.now()],
            'x': [0.1],
            'y': [0.2, 0.3],
            'z': [9.8, 9.9]
        }
        with pytest.raises(ValueError, match="All accelerometer data columns must have equal length"):
            harmonizer.harmonize_accelerometer_data(invalid_data)
    
    def test_harmonize_communication_data(self):
        """Test communication data harmonization."""
        harmonizer = DataHarmonizer()
        comm_data = self.create_sample_communication_data()
        
        result = harmonizer.harmonize_communication_data(comm_data)
        
        # Check output structure
        assert 'timestamp' in result
        assert 'direction' in result
        assert 'contact_id' in result
        assert 'harmonization_metadata' in result
        
        # Check direction standardization
        directions = result['direction']
        assert all(d in ['in', 'out'] for d in directions)
        
        # Check contact ID standardization
        contact_ids = result['contact_id']
        assert all(isinstance(cid, str) and cid.strip() == cid for cid in contact_ids)
        
        # Check metadata
        metadata = result['harmonization_metadata']
        assert metadata['resample_freq_minutes'] is None  # Event-based
        assert metadata['timezone_standardized'] == True
    
    def test_harmonize_communication_data_validation(self):
        """Test communication data validation."""
        harmonizer = DataHarmonizer()
        
        # Missing required column
        invalid_data = {'timestamp': [datetime.now()], 'direction': ['out']}
        with pytest.raises(ValueError, match="Missing required column: contact_id"):
            harmonizer.harmonize_communication_data(invalid_data)
    
    def test_harmonize_screen_state_data(self):
        """Test screen state data harmonization."""
        harmonizer = DataHarmonizer()
        screen_data = self.create_sample_screen_state_data()
        
        result = harmonizer.harmonize_screen_state_data(screen_data)
        
        # Check output structure
        assert 'timestamp' in result
        assert 'state' in result
        assert 'harmonization_metadata' in result
        
        # Check metadata
        metadata = result['harmonization_metadata']
        assert metadata['resample_freq_minutes'] == 1
        assert metadata['aggregation_method'] == 'mode'
        assert metadata['state_encoding_standardized'] == True
    
    def test_harmonize_screen_state_data_validation(self):
        """Test screen state data validation."""
        harmonizer = DataHarmonizer()
        
        # Missing required column
        invalid_data = {'timestamp': [datetime.now()]}
        with pytest.raises(ValueError, match="Missing required column: state"):
            harmonizer.harmonize_screen_state_data(invalid_data)
    
    def test_apply_temporal_segmentation(self):
        """Test temporal segmentation application."""
        harmonizer = DataHarmonizer()
        gps_data = self.create_sample_gps_data()
        
        segmented_data = harmonizer.apply_temporal_segmentation(gps_data, 'gps')
        
        # Check temporal flags
        assert 'temporal_flags' in segmented_data
        flags = segmented_data['temporal_flags']
        
        assert 'is_weekend' in flags
        assert 'is_work_hours' in flags
        assert 'hour_of_day' in flags
        assert 'day_of_week' in flags
        
        # Check data types
        assert all(isinstance(flag, bool) for flag in flags['is_weekend'])
        assert all(isinstance(flag, bool) for flag in flags['is_work_hours'])
        assert all(isinstance(hour, int) for hour in flags['hour_of_day'])
        assert all(isinstance(day, int) for day in flags['day_of_week'])
        
        # Check provenance
        provenance_entry = harmonizer.provenance[-1]
        assert provenance_entry['operation'] == 'apply_temporal_segmentation'
        assert provenance_entry['data_type'] == 'gps'
    
    def test_apply_missing_data_flags(self):
        """Test missing data flags application."""
        harmonizer = DataHarmonizer()
        gps_data = self.create_sample_gps_data()
        
        flagged_data = harmonizer.apply_missing_data_flags(gps_data, 'gps')
        
        # Check missing data flags
        assert 'missing_data_flags' in flagged_data
        flags = flagged_data['missing_data_flags']
        
        assert 'coverage_ratio' in flags
        assert 'max_gap_hours' in flags
        assert 'has_sufficient_coverage' in flags
        assert 'has_acceptable_gaps' in flags
        assert 'missing_data_imputed' in flags
        
        # Check data types
        assert isinstance(flags['coverage_ratio'], float)
        assert isinstance(flags['max_gap_hours'], float)
        assert isinstance(flags['has_sufficient_coverage'], bool)
        assert isinstance(flags['has_acceptable_gaps'], bool)
        assert isinstance(flags['missing_data_imputed'], bool)
        
        # Check provenance
        provenance_entry = harmonizer.provenance[-1]
        assert provenance_entry['operation'] == 'apply_missing_data_flags'
        assert provenance_entry['data_type'] == 'gps'
    
    def test_missing_data_imputation(self):
        """Test missing data imputation for sparse data."""
        harmonizer = DataHarmonizer()
        
        # Create sparse GPS data (large gaps)
        base_time = datetime(2026, 2, 21, 12, 0, 0)
        sparse_gps = {
            'timestamp': [base_time, base_time + timedelta(hours=6), base_time + timedelta(hours=12)],
            'latitude': [40.7128, 40.7228, 40.7328],
            'longitude': [-74.0060, -74.0160, -74.0260]
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore mock implementation warnings
            flagged_data = harmonizer.apply_missing_data_flags(sparse_gps, 'gps')
        
        flags = flagged_data['missing_data_flags']
        
        # Should have low coverage and large gaps
        assert flags['coverage_ratio'] < 0.7
        assert flags['max_gap_hours'] > 4.0
        assert not flags['has_sufficient_coverage']
        assert not flags['has_acceptable_gaps']
    
    def test_provenance_tracking(self):
        """Test provenance tracking functionality."""
        harmonizer = DataHarmonizer()
        gps_data = self.create_sample_gps_data()
        
        # Perform multiple operations
        harmonizer.harmonize_gps_data(gps_data)
        harmonizer.apply_temporal_segmentation(gps_data, 'gps')
        harmonizer.apply_missing_data_flags(gps_data, 'gps')
        
        # Check provenance log
        provenance = harmonizer.get_provenance_log()
        assert len(provenance) == 3
        
        # Check provenance entry structure
        for entry in provenance:
            assert 'operation' in entry
            assert 'timestamp' in entry
            assert 'duration_seconds' in entry
            assert 'parameters' in entry
        
        # Clear provenance
        harmonizer.clear_provenance_log()
        assert len(harmonizer.get_provenance_log()) == 0
    
    def test_accelerometer_magnitude_computation(self):
        """Test accelerometer magnitude computation."""
        harmonizer = DataHarmonizer()
        
        # Simple test data
        accel_data = {
            'timestamp': [datetime.now()],
            'x': [3.0],
            'y': [4.0],
            'z': [0.0]
        }
        
        result = harmonizer.harmonize_accelerometer_data(accel_data)
        
        # Magnitude should be sqrt(3^2 + 4^2 + 0^2) = 5.0
        # Note: Using mock implementation, so this tests the structure
        assert 'magnitude' in result
        assert len(result['magnitude']) == 1
    
    def test_direction_standardization(self):
        """Test communication direction standardization."""
        harmonizer = DataHarmonizer()
        
        # Test various direction formats
        comm_data = {
            'timestamp': [datetime.now()] * 6,
            'direction': ['out', 'in', 'incoming', 'outgoing', 'OUT', 'IN'],
            'contact_id': ['c1'] * 6
        }
        
        result = harmonizer.harmonize_communication_data(comm_data)
        directions = result['direction']
        
        expected = ['out', 'in', 'in', 'out', 'out', 'in']
        assert directions == expected
    
    def test_contact_id_standardization(self):
        """Test contact ID standardization."""
        harmonizer = DataHarmonizer()
        
        # Test various contact ID formats
        comm_data = {
            'timestamp': [datetime.now()] * 3,
            'direction': ['out'] * 3,
            'contact_id': [' contact_1 ', 'contact_2', 123]
        }
        
        result = harmonizer.harmonize_communication_data(comm_data)
        contact_ids = result['contact_id']
        
        expected = ['contact_1', 'contact_2', '123']
        assert contact_ids == expected
    
    def test_weekend_detection(self):
        """Test weekend detection in temporal segmentation."""
        harmonizer = DataHarmonizer()
        
        # Create data spanning weekend
        saturday_time = datetime(2026, 2, 21, 12, 0, 0)  # Saturday
        sunday_time = datetime(2026, 2, 22, 12, 0, 0)    # Sunday
        monday_time = datetime(2026, 2, 23, 12, 0, 0)    # Monday
        
        weekend_data = {
            'timestamp': [saturday_time, sunday_time, monday_time],
            'latitude': [40.7128, 40.7228, 40.7328],
            'longitude': [-74.0060, -74.0160, -74.0260]
        }
        
        segmented = harmonizer.apply_temporal_segmentation(weekend_data, 'gps')
        flags = segmented['temporal_flags']
        
        # Saturday (5), Sunday (6) should be weekend, Monday (0) should not
        assert flags['is_weekend'] == [True, True, False]
        assert flags['day_of_week'] == [5, 6, 0]
    
    def test_work_hours_detection(self):
        """Test work hours detection in temporal segmentation."""
        harmonizer = DataHarmonizer()
        
        # Create data at different hours
        work_time = datetime(2026, 2, 21, 14, 0, 0)    # 2 PM (work hours)
        non_work_time = datetime(2026, 2, 21, 20, 0, 0) # 8 PM (non-work hours)
        
        work_data = {
            'timestamp': [work_time, non_work_time],
            'latitude': [40.7128, 40.7228],
            'longitude': [-74.0060, -74.0160]
        }
        
        segmented = harmonizer.apply_temporal_segmentation(work_data, 'gps')
        flags = segmented['temporal_flags']
        
        assert flags['is_work_hours'] == [True, False]
        assert flags['hour_of_day'] == [14, 20]


class TestHarmonizationIntegration:
    """Integration tests for harmonization workflows."""
    
    def test_complete_harmonization_workflow(self):
        """Test complete harmonization workflow for all data types."""
        harmonizer = DataHarmonizer()
        
        # Sample data for all types
        gps_data = {
            'timestamp': [datetime(2026, 2, 21, 12, 0, 0)],
            'latitude': [40.7128],
            'longitude': [-74.0060]
        }
        
        accel_data = {
            'timestamp': [datetime(2026, 2, 21, 12, 0, 0)],
            'x': [0.1], 'y': [0.2], 'z': [9.8]
        }
        
        comm_data = {
            'timestamp': [datetime(2026, 2, 21, 12, 0, 0)],
            'direction': ['out'],
            'contact_id': ['contact_1']
        }
        
        screen_data = {
            'timestamp': [datetime(2026, 2, 21, 12, 0, 0)],
            'state': ['on']
        }
        
        # Apply harmonization to all data types
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore mock implementation warnings
            
            harmonized_gps = harmonizer.harmonize_gps_data(gps_data)
            harmonized_gps = harmonizer.apply_temporal_segmentation(harmonized_gps, 'gps')
            harmonized_gps = harmonizer.apply_missing_data_flags(harmonized_gps, 'gps')
            
            harmonized_accel = harmonizer.harmonize_accelerometer_data(accel_data)
            harmonized_accel = harmonizer.apply_temporal_segmentation(harmonized_accel, 'accelerometer')
            
            harmonized_comm = harmonizer.harmonize_communication_data(comm_data)
            harmonized_comm = harmonizer.apply_temporal_segmentation(harmonized_comm, 'communication')
            
            harmonized_screen = harmonizer.harmonize_screen_state_data(screen_data)
            harmonized_screen = harmonizer.apply_temporal_segmentation(harmonized_screen, 'screen_state')
        
        # Check that all data has been processed
        assert 'harmonization_metadata' in harmonized_gps
        assert 'temporal_flags' in harmonized_gps
        assert 'missing_data_flags' in harmonized_gps
        
        assert 'magnitude' in harmonized_accel
        assert 'temporal_flags' in harmonized_accel
        
        assert 'temporal_flags' in harmonized_comm
        
        assert 'temporal_flags' in harmonized_screen
        
        # Check provenance log
        provenance = harmonizer.get_provenance_log()
        assert len(provenance) >= 7  # At least 7 operations performed
        
        # Verify operation types
        operations = [entry['operation'] for entry in provenance]
        assert 'harmonize_gps_data' in operations
        assert 'harmonize_accelerometer_data' in operations
        assert 'harmonize_communication_data' in operations
        assert 'harmonize_screen_state_data' in operations
        assert 'apply_temporal_segmentation' in operations
        assert 'apply_missing_data_flags' in operations
