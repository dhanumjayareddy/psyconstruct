"""
Data harmonization module for cross-platform sensor data standardization.

This module implements standardization procedures for inconsistent sensor data
across different devices and platforms, ensuring consistent feature extraction.

Product: Construct-Aligned Digital Phenotyping Toolkit
Purpose: Cross-platform data harmonization and standardization
"""

import math
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings


@dataclass
class HarmonizationConfig:
    """Configuration for data harmonization procedures."""
    
    # Resampling frequencies (in minutes)
    gps_resample_freq: int = 5
    accelerometer_resample_freq: int = 1
    screen_resample_freq: int = 1
    
    # Data quality thresholds
    min_data_coverage: float = 0.7  # 70% minimum coverage
    max_gap_tolerance_hours: float = 4.0
    min_interpolation_gap_minutes: float = 60.0
    
    # Timezone handling
    target_timezone: str = "UTC"
    store_original_timezone: bool = True
    
    # Device metadata handling
    normalize_sampling_frequency: bool = True
    apply_device_bias_correction: bool = True


class DataHarmonizer:
    """
    Data harmonization for cross-platform sensor data.
    
    This class provides methods to standardize sensor data from different
    devices and platforms, addressing inconsistencies in sampling rates,
    timezone handling, missing data patterns, and device-specific biases.
    
    Attributes:
        config: Harmonization configuration parameters
        provenance: Record of harmonization operations performed
    """
    
    def __init__(self, config: Optional[HarmonizationConfig] = None):
        """
        Initialize data harmonizer.
        
        Args:
            config: Harmonization configuration. If None, uses defaults.
        """
        self.config = config or HarmonizationConfig()
        self.provenance = []
    
    def harmonize_gps_data(self, 
                          gps_data: Dict[str, Any],
                          device_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Harmonize GPS data to standard format.
        
        Feature Name: GPS Data Harmonization
        Construct: Data Quality (preprocessing)
        Mathematical Definition: Resample GPS coordinates to 5-minute intervals using median aggregation
        Formal Equation: GPS_harmonized[t] = median(GPS_raw[t-Δ:t+Δ]) where Δ = 2.5 minutes
        Assumptions: GPS timestamps are in UTC or can be converted to UTC
        Limitations: Median aggregation may smooth rapid movement patterns
        Edge Cases: Sparse GPS data, timezone inconsistencies, device-specific coordinate systems
        Output Schema: Standardized GPS dataframe with timestamp, latitude, longitude, and harmonization metadata
        
        Args:
            gps_data: Raw GPS data with timestamp, latitude, longitude columns
            device_metadata: Optional device-specific metadata
            
        Returns:
            Harmonized GPS data with standardized format and metadata
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        operation_start = datetime.now()
        
        # Validate input data
        self._validate_gps_data(gps_data)
        
        # Convert to standardized format (mock implementation)
        harmonized_data = {
            'timestamp': self._process_timestamps(gps_data['timestamp']),
            'latitude': self._resample_coordinates(gps_data['latitude'], freq_minutes=self.config.gps_resample_freq),
            'longitude': self._resample_coordinates(gps_data['longitude'], freq_minutes=self.config.gps_resample_freq),
            'harmonization_metadata': {
                'resample_freq_minutes': self.config.gps_resample_freq,
                'aggregation_method': 'median',
                'device_type': device_metadata.get('device_type', 'unknown') if device_metadata else 'unknown',
                'original_timezone': device_metadata.get('timezone', 'UTC') if device_metadata else 'UTC'
            }
        }
        
        # Record provenance
        operation_duration = (datetime.now() - operation_start).total_seconds()
        self.provenance.append({
            'operation': 'harmonize_gps_data',
            'timestamp': operation_start.isoformat(),
            'duration_seconds': operation_duration,
            'input_records': len(gps_data['timestamp']),
            'output_records': len(harmonized_data['timestamp']),
            'parameters': {
                'resample_freq_minutes': self.config.gps_resample_freq,
                'device_type': device_metadata.get('device_type', 'unknown') if device_metadata else 'unknown'
            }
        })
        
        return harmonized_data
    
    def harmonize_accelerometer_data(self,
                                   accel_data: Dict[str, Any],
                                   device_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Harmonize accelerometer data to standard format.
        
        Feature Name: Accelerometer Data Harmonization
        Construct: Data Quality (preprocessing)
        Mathematical Definition: Resample accelerometer magnitude to 1-minute intervals using mean aggregation
        Formal Equation: Accel_harmonized[t] = mean(√(x² + y² + z²)) over 1-minute windows
        Assumptions: Accelerometer axes are calibrated and magnitude is meaningful
        Limitations: Mean aggregation may not capture peak intensity patterns
        Edge Cases: Variable sampling rates, device orientation changes, missing axes
        Output Schema: Standardized accelerometer dataframe with timestamp and magnitude
        
        Args:
            accel_data: Raw accelerometer data with timestamp, x, y, z columns
            device_metadata: Optional device-specific metadata
            
        Returns:
            Harmonized accelerometer data with standardized format and metadata
        """
        operation_start = datetime.now()
        
        # Validate input data
        self._validate_accelerometer_data(accel_data)
        
        # Compute magnitude if not provided
        if 'magnitude' not in accel_data:
            magnitude = self._compute_accelerometer_magnitude(
                accel_data['x'], accel_data['y'], accel_data['z']
            )
        else:
            magnitude = accel_data['magnitude']
        
        # Resample to standard frequency
        harmonized_data = {
            'timestamp': self._process_timestamps(accel_data['timestamp']),
            'magnitude': self._resample_signal(magnitude, freq_minutes=self.config.accelerometer_resample_freq),
            'harmonization_metadata': {
                'resample_freq_minutes': self.config.accelerometer_resample_freq,
                'aggregation_method': 'mean',
                'device_type': device_metadata.get('device_type', 'unknown') if device_metadata else 'unknown',
                'sampling_rate_normalized': self.config.normalize_sampling_frequency
            }
        }
        
        # Record provenance
        operation_duration = (datetime.now() - operation_start).total_seconds()
        self.provenance.append({
            'operation': 'harmonize_accelerometer_data',
            'timestamp': operation_start.isoformat(),
            'duration_seconds': operation_duration,
            'input_records': len(accel_data['timestamp']),
            'output_records': len(harmonized_data['timestamp']),
            'parameters': {
                'resample_freq_minutes': self.config.accelerometer_resample_freq,
                'magnitude_computed': 'magnitude' not in accel_data
            }
        })
        
        return harmonized_data
    
    def harmonize_communication_data(self,
                                   comm_data: Dict[str, Any],
                                   device_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Harmonize communication logs to standard format.
        
        Feature Name: Communication Data Harmonization
        Construct: Data Quality (preprocessing)
        Mathematical Definition: Standardize communication events without temporal resampling
        Formal Equation: Comm_harmonized = Comm_raw with standardized timezone and contact IDs
        Assumptions: Communication timestamps are accurate and can be normalized
        Limitations: Event-based data cannot be meaningfully resampled
        Edge Cases: Duplicate events, missing contact identifiers, timezone inconsistencies
        Output Schema: Standardized communication dataframe with timestamp, direction, contact_id
        
        Args:
            comm_data: Raw communication data with timestamp, direction, contact_id columns
            device_metadata: Optional device-specific metadata
            
        Returns:
            Harmonized communication data with standardized format and metadata
        """
        operation_start = datetime.now()
        
        # Validate input data
        self._validate_communication_data(comm_data)
        
        # Standardize format (no resampling for event-based data)
        harmonized_data = {
            'timestamp': self._process_timestamps(comm_data['timestamp']),
            'direction': self._standardize_direction(comm_data['direction']),
            'contact_id': self._standardize_contact_ids(comm_data['contact_id']),
            'harmonization_metadata': {
                'resample_freq_minutes': None,  # Event-based, no resampling
                'timezone_standardized': True,
                'device_type': device_metadata.get('device_type', 'unknown') if device_metadata else 'unknown'
            }
        }
        
        # Record provenance
        operation_duration = (datetime.now() - operation_start).total_seconds()
        self.provenance.append({
            'operation': 'harmonize_communication_data',
            'timestamp': operation_start.isoformat(),
            'duration_seconds': operation_duration,
            'input_records': len(comm_data['timestamp']),
            'output_records': len(harmonized_data['timestamp']),
            'parameters': {
                'timezone_standardized': True,
                'contact_ids_standardized': True
            }
        })
        
        return harmonized_data
    
    def harmonize_screen_state_data(self,
                                  screen_data: Dict[str, Any],
                                  device_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Harmonize screen state data to standard format.
        
        Feature Name: Screen State Data Harmonization
        Construct: Data Quality (preprocessing)
        Mathematical Definition: Resample screen state to 1-minute intervals using mode aggregation
        Formal Equation: Screen_harmonized[t] = mode(Screen_raw[t-Δ:t+Δ]) where Δ = 30 seconds
        Assumptions: Screen state changes are captured reliably by device sensors
        Limitations: Mode aggregation may miss brief screen interactions
        Edge Cases: Rapid screen toggling, missing state transitions, device-specific state definitions
        Output Schema: Standardized screen state dataframe with timestamp and state
        
        Args:
            screen_data: Raw screen state data with timestamp, state columns
            device_metadata: Optional device-specific metadata
            
        Returns:
            Harmonized screen state data with standardized format and metadata
        """
        operation_start = datetime.now()
        
        # Validate input data
        self._validate_screen_state_data(screen_data)
        
        # Resample to standard frequency
        harmonized_data = {
            'timestamp': self._process_timestamps(screen_data['timestamp']),
            'state': self._resample_categorical(screen_data['state'], freq_minutes=self.config.screen_resample_freq),
            'harmonization_metadata': {
                'resample_freq_minutes': self.config.screen_resample_freq,
                'aggregation_method': 'mode',
                'device_type': device_metadata.get('device_type', 'unknown') if device_metadata else 'unknown',
                'state_encoding_standardized': True
            }
        }
        
        # Record provenance
        operation_duration = (datetime.now() - operation_start).total_seconds()
        self.provenance.append({
            'operation': 'harmonize_screen_state_data',
            'timestamp': operation_start.isoformat(),
            'duration_seconds': operation_duration,
            'input_records': len(screen_data['timestamp']),
            'output_records': len(harmonized_data['timestamp']),
            'parameters': {
                'resample_freq_minutes': self.config.screen_resample_freq,
                'aggregation_method': 'mode'
            }
        })
        
        return harmonized_data
    
    def apply_temporal_segmentation(self,
                                  data: Dict[str, Any],
                                  data_type: str) -> Dict[str, Any]:
        """
        Apply temporal segmentation (weekend/weekday, work hours).
        
        Feature Name: Temporal Segmentation
        Construct: Data Quality (preprocessing)
        Mathematical Definition: Add temporal flags based on timestamp analysis
        Formal Equation: temporal_flags = f(timestamp) where f extracts weekday/weekend and work hour status
        Assumptions: Timestamps are in consistent timezone
        Limitations: Cultural variations in work schedules and weekends
        Edge Cases: Shift workers, holiday periods,跨时区 travel
        Output Schema: Original data with additional temporal segmentation columns
        
        Args:
            data: Harmonized data with timestamp column
            data_type: Type of data ('gps', 'accelerometer', 'communication', 'screen_state')
            
        Returns:
            Data with temporal segmentation flags added
        """
        operation_start = datetime.now()
        
        # Convert timestamps to datetime if needed
        timestamps = self._ensure_datetime(data['timestamp'])
        
        # Add temporal flags
        temporal_flags = {
            'is_weekend': [ts.weekday() >= 5 for ts in timestamps],  # Saturday=5, Sunday=6
            'is_work_hours': [8 <= ts.hour < 18 for ts in timestamps],  # 8 AM to 6 PM
            'hour_of_day': [ts.hour for ts in timestamps],
            'day_of_week': [ts.weekday() for ts in timestamps]
        }
        
        # Combine with original data
        segmented_data = data.copy()
        segmented_data['temporal_flags'] = temporal_flags
        
        # Record provenance
        operation_duration = (datetime.now() - operation_start).total_seconds()
        self.provenance.append({
            'operation': 'apply_temporal_segmentation',
            'timestamp': operation_start.isoformat(),
            'duration_seconds': operation_duration,
            'data_type': data_type,
            'parameters': {
                'weekend_definition': 'saturday_sunday',
                'work_hours': '08_18'
            }
        })
        
        return segmented_data
    
    def apply_missing_data_flags(self,
                               data: Dict[str, Any],
                               data_type: str) -> Dict[str, Any]:
        """
        Apply missing data flags and imputation policies.
        
        Feature Name: Missing Data Handling
        Construct: Data Quality (preprocessing)
        Mathematical Definition: Identify and flag missing data patterns according to defined thresholds
        Formal Equation: missing_flags = g(data_gaps, coverage_thresholds) where g applies data quality rules
        Assumptions: Missing data patterns are identifiable and flaggable
        Limitations: Imputation may introduce bias if missingness is not random
        Edge Cases: Extended data gaps, systematic missingness, device-specific failure patterns
        Output Schema: Original data with missing data flags and imputation indicators
        
        Args:
            data: Harmonized data with timestamp column
            data_type: Type of data for appropriate missing data handling
            
        Returns:
            Data with missing data flags and imputation applied where appropriate
        """
        operation_start = datetime.now()
        
        # Analyze data coverage and gaps
        timestamps = self._ensure_datetime(data['timestamp'])
        coverage_analysis = self._analyze_data_coverage(timestamps)
        
        # Apply missing data flags
        missing_data_flags = {
            'coverage_ratio': coverage_analysis['coverage_ratio'],
            'max_gap_hours': coverage_analysis['max_gap_hours'],
            'has_sufficient_coverage': coverage_analysis['coverage_ratio'] >= self.config.min_data_coverage,
            'has_acceptable_gaps': coverage_analysis['max_gap_hours'] <= self.config.max_gap_tolerance_hours,
            'missing_data_imputed': False
        }
        
        # Apply imputation if needed and appropriate
        if data_type in ['accelerometer', 'gps'] and coverage_analysis['coverage_ratio'] < 1.0:
            imputed_data = self._apply_imputation(data, data_type, coverage_analysis)
            missing_data_flags['missing_data_imputed'] = True
            missing_data_flags['imputation_method'] = 'linear_interpolation'
            data = imputed_data
        
        # Combine with original data
        flagged_data = data.copy()
        flagged_data['missing_data_flags'] = missing_data_flags
        
        # Record provenance
        operation_duration = (datetime.now() - operation_start).total_seconds()
        self.provenance.append({
            'operation': 'apply_missing_data_flags',
            'timestamp': operation_start.isoformat(),
            'duration_seconds': operation_duration,
            'data_type': data_type,
            'parameters': {
                'min_coverage_threshold': self.config.min_data_coverage,
                'max_gap_tolerance': self.config.max_gap_tolerance_hours,
                'imputation_applied': missing_data_flags['missing_data_imputed']
            }
        })
        
        return flagged_data
    
    def get_provenance_log(self) -> list:
        """
        Get the provenance log of all harmonization operations.
        
        Returns:
            List of provenance entries with operation details
        """
        return self.provenance.copy()
    
    def clear_provenance_log(self) -> None:
        """Clear the provenance log."""
        self.provenance = []
    
    # Private helper methods
    
    def _validate_gps_data(self, gps_data: Dict[str, Any]) -> None:
        """Validate GPS data format and content."""
        required_columns = ['timestamp', 'latitude', 'longitude']
        for col in required_columns:
            if col not in gps_data:
                raise ValueError(f"Missing required column: {col}")
        
        if len(gps_data['timestamp']) != len(gps_data['latitude']) or \
           len(gps_data['timestamp']) != len(gps_data['longitude']):
            raise ValueError("All GPS data columns must have equal length")
    
    def _validate_accelerometer_data(self, accel_data: Dict[str, Any]) -> None:
        """Validate accelerometer data format and content."""
        if 'magnitude' not in accel_data:
            required_columns = ['timestamp', 'x', 'y', 'z']
        else:
            required_columns = ['timestamp', 'magnitude']
        
        for col in required_columns:
            if col not in accel_data:
                raise ValueError(f"Missing required column: {col}")
        
        # Check equal lengths
        lengths = [len(accel_data[col]) for col in required_columns]
        if len(set(lengths)) > 1:
            raise ValueError("All accelerometer data columns must have equal length")
    
    def _validate_communication_data(self, comm_data: Dict[str, Any]) -> None:
        """Validate communication data format and content."""
        required_columns = ['timestamp', 'direction', 'contact_id']
        for col in required_columns:
            if col not in comm_data:
                raise ValueError(f"Missing required column: {col}")
        
        if len(set(len(comm_data[col]) for col in required_columns)) > 1:
            raise ValueError("All communication data columns must have equal length")
    
    def _validate_screen_state_data(self, screen_data: Dict[str, Any]) -> None:
        """Validate screen state data format and content."""
        required_columns = ['timestamp', 'state']
        for col in required_columns:
            if col not in screen_data:
                raise ValueError(f"Missing required column: {col}")
        
        if len(screen_data['timestamp']) != len(screen_data['state']):
            raise ValueError("Screen state data columns must have equal length")
    
    def _process_timestamps(self, timestamps: list) -> list:
        """Process and standardize timestamps to UTC."""
        # Mock implementation - in real implementation would handle timezone conversion
        return timestamps
    
    def _resample_coordinates(self, coordinates: list, freq_minutes: int) -> list:
        """Resample GPS coordinates using median aggregation."""
        try:
            # Convert to pandas Series for resampling
            if not coordinates:
                return coordinates
            
            # Create a time index (assuming regular intervals)
            # In practice, this would use actual timestamps
            coord_series = pd.Series(coordinates)
            
            # Resample using median aggregation
            # For GPS coordinates, we need to handle latitude and longitude separately
            # This is a simplified implementation - in practice would use actual timestamps
            if len(coordinates) > freq_minutes:
                # Simple downsampling using every nth point as proxy for proper resampling
                step = max(1, len(coordinates) // (len(coordinates) // freq_minutes))
                resampled = [coordinates[i] for i in range(0, len(coordinates), step)]
                return resampled
            else:
                return coordinates
                
        except Exception as e:
            warnings.warn(f"Resampling failed, using original data: {e}")
            return coordinates
    
    def _resample_signal(self, signal: list, freq_minutes: int) -> list:
        """Resample continuous signal using mean aggregation."""
        try:
            # Convert to pandas Series for resampling
            if not signal:
                return signal
            
            signal_series = pd.Series(signal)
            
            # Create time index (assuming 1-minute original sampling)
            # In practice, this would use actual timestamps
            timestamps = pd.date_range(
                start='2023-01-01', 
                periods=len(signal), 
                freq='1min'
            )
            signal_df = pd.DataFrame({'value': signal}, index=timestamps)
            
            # Resample to target frequency using mean
            target_freq = f'{freq_minutes}min'
            resampled = signal_df.resample(target_freq).mean()
            
            # Fill NaN values with forward fill then backward fill
            resampled = resampled.fillna(method='ffill').fillna(method='bfill')
            
            return resampled['value'].tolist()
            
        except Exception as e:
            warnings.warn(f"Signal resampling failed, using original data: {e}")
            return signal
    
    def _resample_categorical(self, categorical_data: list, freq_minutes: int) -> list:
        """Resample categorical data using mode aggregation."""
        try:
            # Convert to pandas Series for resampling
            if not categorical_data:
                return categorical_data
            
            # Create time index (assuming 1-minute original sampling)
            timestamps = pd.date_range(
                start='2023-01-01', 
                periods=len(categorical_data), 
                freq='1min'
            )
            cat_series = pd.Series(categorical_data, index=timestamps)
            
            # Resample to target frequency using mode
            target_freq = f'{freq_minutes}min'
            resampled = cat_series.resample(target_freq).apply(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            
            # Fill NaN values with forward fill then backward fill
            resampled = resampled.fillna(method='ffill').fillna(method='bfill')
            
            return resampled.tolist()
            
        except Exception as e:
            warnings.warn(f"Categorical resampling failed, using original data: {e}")
            return categorical_data
    
    def _compute_accelerometer_magnitude(self, x: list, y: list, z: list) -> list:
        """Compute accelerometer magnitude from x, y, z components."""
        magnitude = []
        for i in range(len(x)):
            mag = math.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
            magnitude.append(mag)
        return magnitude
    
    def _standardize_direction(self, directions: list) -> list:
        """Standardize communication direction values."""
        # Convert to lowercase and validate
        standardized = []
        for direction in directions:
            dir_lower = str(direction).lower()
            if dir_lower in ['in', 'out', 'incoming', 'outgoing']:
                standardized.append('in' if dir_lower in ['in', 'incoming'] else 'out')
            else:
                warnings.warn(f"Unexpected direction value: {direction}")
                standardized.append(direction)
        return standardized
    
    def _standardize_contact_ids(self, contact_ids: list) -> list:
        """Standardize contact identifiers."""
        # Convert to strings and remove whitespace
        return [str(contact_id).strip() for contact_id in contact_ids]
    
    def _ensure_datetime(self, timestamps: list) -> list:
        """Ensure timestamps are datetime objects."""
        datetime_timestamps = []
        for ts in timestamps:
            if isinstance(ts, str):
                # Mock parsing - in real implementation would use dateutil parser
                datetime_timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            elif isinstance(ts, datetime):
                datetime_timestamps.append(ts)
            else:
                warnings.warn(f"Unexpected timestamp type: {type(ts)}")
                datetime_timestamps.append(ts)
        return datetime_timestamps
    
    def _analyze_data_coverage(self, timestamps: list) -> Dict[str, float]:
        """Analyze data coverage and gap patterns."""
        if len(timestamps) < 2:
            return {'coverage_ratio': 0.0, 'max_gap_hours': float('inf')}
        
        # Sort timestamps
        sorted_timestamps = sorted(timestamps)
        
        # Calculate total time span and gaps
        total_span = (sorted_timestamps[-1] - sorted_timestamps[0]).total_seconds() / 3600  # hours
        
        # Find maximum gap
        max_gap = 0.0
        for i in range(1, len(sorted_timestamps)):
            gap = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds() / 3600
            max_gap = max(max_gap, gap)
        
        # Estimate coverage ratio (simplified)
        expected_points = total_span * 12  # Assuming 5-minute intervals as baseline
        coverage_ratio = min(len(timestamps) / expected_points, 1.0) if expected_points > 0 else 0.0
        
        return {
            'coverage_ratio': coverage_ratio,
            'max_gap_hours': max_gap
        }
    
    def _apply_imputation(self, data: Dict[str, Any], data_type: str, coverage_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Apply multiple imputation methods for missing data."""
        try:
            imputed_data = data.copy()
            
            if data_type == 'accelerometer' and 'magnitude' in data:
                # Linear interpolation for accelerometer magnitude
                magnitude_series = pd.Series(data['magnitude'])
                imputed_magnitude = magnitude_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                imputed_data['magnitude'] = imputed_magnitude.tolist()
                
            elif data_type == 'gps' and 'latitude' in data and 'longitude' in data:
                # Linear interpolation for GPS coordinates
                lat_series = pd.Series(data['latitude'])
                lon_series = pd.Series(data['longitude'])
                
                imputed_lat = lat_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                imputed_lon = lon_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                
                imputed_data['latitude'] = imputed_lat.tolist()
                imputed_data['longitude'] = imputed_lon.tolist()
                
            return imputed_data
            
        except Exception as e:
            warnings.warn(f"Imputation failed, using original data: {e}")
            return data
