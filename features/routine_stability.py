"""
Routine Stability (RS) construct features.

This module implements features related to routine stability patterns,
including sleep consistency, duration, activity fragmentation, and circadian rhythms.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import math
import statistics


@dataclass
class SleepOnsetConfig:
    """Configuration for sleep onset consistency feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 14  # 2-week analysis for sleep patterns
    min_nights_data: int = 7  # Minimum nights with sleep data
    
    # Sleep detection parameters
    min_screen_off_duration_hours: float = 2.0  # Minimum screen-off to consider as sleep
    max_screen_off_duration_hours: float = 12.0  # Maximum screen-off duration
    screen_off_threshold_minutes: float = 5.0  # Minimum gap to detect screen-off
    
    # Quality requirements
    min_data_coverage: float = 0.7  # 70% minimum coverage
    max_missing_hours_per_day: float = 4.0  # Maximum missing data per day
    
    # Processing options
    outlier_detection: bool = True  # Detect and handle outlier sleep times
    weekend_separation: bool = True  # Analyze weekdays and weekends separately


@dataclass
class SleepDurationConfig:
    """Configuration for sleep duration feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 14  # 2-week analysis
    min_nights_data: int = 7  # Minimum nights with sleep data
    
    # Sleep duration parameters
    min_sleep_duration_hours: float = 3.0  # Minimum plausible sleep duration
    max_sleep_duration_hours: float = 12.0  # Maximum plausible sleep duration
    sleep_detection_method: str = "screen_off"  # "screen_off" or "accelerometer"
    
    # Quality requirements
    min_data_coverage: float = 0.7
    max_missing_hours_per_day: float = 4.0
    
    # Processing options
    calculate_variability: bool = True  # Calculate sleep duration variability
    weekday_weekend_analysis: bool = True


@dataclass
class ActivityFragmentationConfig:
    """Configuration for activity fragmentation feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 7  # Weekly analysis
    min_days_data: int = 5  # Minimum days with activity data
    
    # Fragmentation parameters
    time_resolution_minutes: int = 60  # Hourly bins for activity distribution
    activity_threshold: float = 0.1  # Minimum activity to count as active
    entropy_base: str = "natural"  # "natural" or "log2"
    
    # Quality requirements
    min_data_coverage: float = 0.8  # High coverage needed for fragmentation
    min_active_hours_per_day: int = 4  # Minimum active hours for reliable entropy
    
    # Processing options
    normalize_by_total_activity: bool = True  # Normalize activity before entropy calculation
    smooth_activity_distribution: bool = False  # Apply smoothing to hourly distribution


@dataclass
class CircadianMidpointConfig:
    """Configuration for circadian midpoint feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 14  # 2-week analysis
    min_nights_data: int = 7  # Minimum nights with sleep/wake data
    
    # Midpoint calculation parameters
    sleep_detection_method: str = "screen_off"  # "screen_off" or "accelerometer"
    wake_detection_method: str = "screen_on"  # "screen_on" or "accelerometer"
    
    # Quality requirements
    min_data_coverage: float = 0.7
    max_missing_hours_per_day: float = 4.0
    
    # Processing options
    calculate_phase_shift: bool = True  # Calculate day-to-day phase shifts
    weekend_analysis: bool = True  # Separate weekday/weekend analysis
    outlier_detection: bool = True  # Detect outlier midpoints


class RoutineStabilityFeatures:
    """
    Implementation of Routine Stability (RS) construct features.
    
    This class provides methods for extracting features related to routine
    stability, which reflects the consistency and regularity of daily
    behavioral patterns including sleep, activity, and circadian rhythms.
    
    Attributes:
        sleep_onset_config: Configuration for sleep onset consistency features
        sleep_duration_config: Configuration for sleep duration features
        fragmentation_config: Configuration for activity fragmentation features
        circadian_config: Configuration for circadian midpoint features
        provenance_tracker: Provenance tracking instance
    """
    
    def __init__(self, 
                 sleep_onset_config: Optional[SleepOnsetConfig] = None,
                 sleep_duration_config: Optional[SleepDurationConfig] = None,
                 fragmentation_config: Optional[ActivityFragmentationConfig] = None,
                 circadian_config: Optional[CircadianMidpointConfig] = None):
        """
        Initialize routine stability features extractor.
        
        Args:
            sleep_onset_config: Configuration for sleep onset consistency features
            sleep_duration_config: Configuration for sleep duration features
            fragmentation_config: Configuration for activity fragmentation features
            circadian_config: Configuration for circadian midpoint features
        """
        self.sleep_onset_config = sleep_onset_config or SleepOnsetConfig()
        self.sleep_duration_config = sleep_duration_config or SleepDurationConfig()
        self.fragmentation_config = fragmentation_config or ActivityFragmentationConfig()
        self.circadian_config = circadian_config or CircadianMidpointConfig()
        
        # Import provenance tracker locally to avoid circular imports
        try:
            from ..utils.provenance import get_provenance_tracker
            self.provenance_tracker = get_provenance_tracker()
        except ImportError:
            self.provenance_tracker = None
    
    def sleep_onset_consistency(self,
                               screen_data: Dict[str, Any],
                               window_start: Optional[datetime] = None,
                               window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate sleep onset consistency from screen usage data.
        
        Feature Name: Sleep Onset Consistency
        Construct: Routine Stability (RS)
        Mathematical Definition: Standard deviation of sleep onset times across days
        Formal Equation: SOC = σ(sleep_onset_i) for i = 1 to N days
        Assumptions: Longest screen-off interval represents sleep period
        Limitations: Sensitive to screen usage patterns and device charging behavior
        Edge Cases: No screen-off intervals, multiple long intervals, irregular patterns
        Output Schema: Sleep onset times with consistency metrics and quality indicators
        
        Args:
            screen_data: Dictionary with 'timestamp', 'screen_state' columns (1=on, 0=off)
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing sleep onset consistency values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_sleep_onset_consistency",
                input_parameters={
                    "analysis_window_days": self.sleep_onset_config.analysis_window_days,
                    "min_screen_off_duration_hours": self.sleep_onset_config.min_screen_off_duration_hours,
                    "screen_off_threshold_minutes": self.sleep_onset_config.screen_off_threshold_minutes,
                    "outlier_detection": self.sleep_onset_config.outlier_detection
                }
            )
        
        try:
            # Validate input data
            self._validate_screen_data(screen_data)
            
            # Extract and preprocess screen data
            timestamps = self._ensure_datetime(screen_data['timestamp'])
            screen_states = screen_data['screen_state']
            
            # Set analysis window
            if window_start is None:
                window_start = min(timestamps)
            if window_end is None:
                window_end = max(timestamps)
            
            # Filter data to analysis window
            filtered_data = self._filter_screen_data_by_window(
                timestamps, screen_states, window_start, window_end
            )
            filtered_timestamps, filtered_screen_states = filtered_data
            
            # Check minimum data requirements
            if len(filtered_timestamps) < 100:  # Minimum data points
                raise ValueError(
                    f"Insufficient screen data: {len(filtered_timestamps)} "
                    f"< 100 minimum points"
                )
            
            # Quality assessment
            quality_metrics = self._assess_sleep_data_quality(
                filtered_timestamps, filtered_screen_states, window_start, window_end
            )
            
            # Detect sleep onset times
            sleep_onset_times = self._detect_sleep_onset_times(
                filtered_timestamps, filtered_screen_states
            )
            
            # Calculate consistency metrics
            consistency_metrics = self._calculate_sleep_onset_consistency(
                sleep_onset_times
            )
            
            # Prepare results
            result = {
                'sleep_onset_consistency': consistency_metrics,
                'sleep_onset_sd_hours': consistency_metrics['sleep_onset_sd_hours'],
                'daily_sleep_onsets': consistency_metrics['daily_sleep_onsets'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.sleep_onset_config.analysis_window_days,
                    'min_screen_off_duration_hours': self.sleep_onset_config.min_screen_off_duration_hours,
                    'screen_off_threshold_minutes': self.sleep_onset_config.screen_off_threshold_minutes,
                    'outlier_detection': self.sleep_onset_config.outlier_detection,
                    'weekend_separation': self.sleep_onset_config.weekend_separation
                },
                'data_summary': {
                    'total_screen_points': len(filtered_timestamps),
                    'nights_analyzed': len(sleep_onset_times),
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'sleep_onset_sd_hours': consistency_metrics['sleep_onset_sd_hours'],
                        'nights_analyzed': len(sleep_onset_times),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="sleep_onset_consistency",
                    construct="routine_stability",
                    input_data_summary={
                        'screen_points': len(filtered_timestamps),
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400,
                        'nights_detected': len(sleep_onset_times)
                    },
                    computation_parameters={
                        'min_screen_off_duration_hours': self.sleep_onset_config.min_screen_off_duration_hours,
                        'outlier_detection': self.sleep_onset_config.outlier_detection
                    },
                    result_summary={
                        'sleep_onset_sd_hours': consistency_metrics['sleep_onset_sd_hours'],
                        'mean_sleep_onset_hour': consistency_metrics['mean_sleep_onset_hour'],
                        'sleep_onset_range_hours': consistency_metrics['sleep_onset_range_hours']
                    },
                    data_quality_metrics=quality_metrics,
                    algorithm_version="1.0.0"
                )
            
            return result
            
        except Exception as e:
            # Record failed operation
            if self.provenance_tracker and operation_id:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': False,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    duration_seconds=0.0
                )
            raise
    
    def sleep_duration(self,
                      screen_data: Dict[str, Any],
                      window_start: Optional[datetime] = None,
                      window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate sleep duration from screen usage data.
        
        Feature Name: Sleep Duration
        Construct: Routine Stability (RS)
        Mathematical Definition: Average length of inferred sleep interval
        Formal Equation: SD = (1/N) * Σ(sleep_duration_i) for i = 1 to N nights
        Assumptions: Screen-off intervals represent sleep periods
        Limitations: Sensitive to device usage patterns and charging behavior
        Edge Cases: No clear sleep intervals, fragmented sleep, very short/long durations
        Output Schema: Sleep duration metrics with variability and quality indicators
        
        Args:
            screen_data: Dictionary with 'timestamp', 'screen_state' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing sleep duration values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_sleep_duration",
                input_parameters={
                    "analysis_window_days": self.sleep_duration_config.analysis_window_days,
                    "min_sleep_duration_hours": self.sleep_duration_config.min_sleep_duration_hours,
                    "max_sleep_duration_hours": self.sleep_duration_config.max_sleep_duration_hours,
                    "calculate_variability": self.sleep_duration_config.calculate_variability
                }
            )
        
        try:
            # Validate input data
            self._validate_screen_data(screen_data)
            
            # Extract and preprocess screen data
            timestamps = self._ensure_datetime(screen_data['timestamp'])
            screen_states = screen_data['screen_state']
            
            # Set analysis window
            if window_start is None:
                window_start = min(timestamps)
            if window_end is None:
                window_end = max(timestamps)
            
            # Filter data to analysis window
            filtered_data = self._filter_screen_data_by_window(
                timestamps, screen_states, window_start, window_end
            )
            filtered_timestamps, filtered_screen_states = filtered_data
            
            # Check minimum data requirements
            if len(filtered_timestamps) < 100:
                raise ValueError(
                    f"Insufficient screen data: {len(filtered_timestamps)} "
                    f"< 100 minimum points"
                )
            
            # Quality assessment
            quality_metrics = self._assess_sleep_data_quality(
                filtered_timestamps, filtered_screen_states, window_start, window_end
            )
            
            # Detect sleep intervals
            sleep_intervals = self._detect_sleep_intervals(
                filtered_timestamps, filtered_screen_states
            )
            
            # Calculate sleep duration metrics
            duration_metrics = self._calculate_sleep_duration_metrics(
                sleep_intervals
            )
            
            # Prepare results
            result = {
                'sleep_duration': duration_metrics,
                'mean_sleep_duration_hours': duration_metrics['mean_sleep_duration_hours'],
                'daily_sleep_durations': duration_metrics['daily_sleep_durations'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.sleep_duration_config.analysis_window_days,
                    'min_sleep_duration_hours': self.sleep_duration_config.min_sleep_duration_hours,
                    'max_sleep_duration_hours': self.sleep_duration_config.max_sleep_duration_hours,
                    'sleep_detection_method': self.sleep_duration_config.sleep_detection_method,
                    'calculate_variability': self.sleep_duration_config.calculate_variability
                },
                'data_summary': {
                    'total_screen_points': len(filtered_timestamps),
                    'nights_analyzed': len(sleep_intervals),
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'mean_sleep_duration_hours': duration_metrics['mean_sleep_duration_hours'],
                        'nights_analyzed': len(sleep_intervals),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="sleep_duration",
                    construct="routine_stability",
                    input_data_summary={
                        'screen_points': len(filtered_timestamps),
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400,
                        'sleep_intervals': len(sleep_intervals)
                    },
                    computation_parameters={
                        'min_sleep_duration_hours': self.sleep_duration_config.min_sleep_duration_hours,
                        'max_sleep_duration_hours': self.sleep_duration_config.max_sleep_duration_hours
                    },
                    result_summary={
                        'mean_sleep_duration_hours': duration_metrics['mean_sleep_duration_hours'],
                        'sleep_duration_sd_hours': duration_metrics.get('sleep_duration_sd_hours', 0),
                        'sleep_efficiency': duration_metrics.get('sleep_efficiency', 0)
                    },
                    data_quality_metrics=quality_metrics,
                    algorithm_version="1.0.0"
                )
            
            return result
            
        except Exception as e:
            # Record failed operation
            if self.provenance_tracker and operation_id:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': False,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    duration_seconds=0.0
                )
            raise
    
    def activity_fragmentation(self,
                              activity_data: Dict[str, Any],
                              window_start: Optional[datetime] = None,
                              window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate activity fragmentation from activity data.
        
        Feature Name: Activity Fragmentation
        Construct: Routine Stability (RS)
        Mathematical Definition: Entropy of hourly activity distribution within a day
        Formal Equation: AF = -Σ(p_i * log(p_i)) where p_i is activity proportion in hour i
        Assumptions: Activity distribution reflects daily routine structure
        Limitations: Sensitive to activity measurement quality and daily patterns
        Edge Cases: No activity data, constant activity, highly fragmented patterns
        Output Schema: Fragmentation entropy with temporal patterns and quality metrics
        
        Args:
            activity_data: Dictionary with 'timestamp', 'activity_level' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing activity fragmentation values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_activity_fragmentation",
                input_parameters={
                    "analysis_window_days": self.fragmentation_config.analysis_window_days,
                    "time_resolution_minutes": self.fragmentation_config.time_resolution_minutes,
                    "activity_threshold": self.fragmentation_config.activity_threshold,
                    "entropy_base": self.fragmentation_config.entropy_base,
                    "normalize_by_total_activity": self.fragmentation_config.normalize_by_total_activity
                }
            )
        
        try:
            # Validate input data
            self._validate_activity_data(activity_data)
            
            # Extract and preprocess activity data
            timestamps = self._ensure_datetime(activity_data['timestamp'])
            activity_levels = activity_data['activity_level']
            
            # Set analysis window
            if window_start is None:
                window_start = min(timestamps)
            if window_end is None:
                window_end = max(timestamps)
            
            # Filter data to analysis window
            filtered_data = self._filter_activity_data_by_window(
                timestamps, activity_levels, window_start, window_end
            )
            filtered_timestamps, filtered_activity_levels = filtered_data
            
            # Check minimum data requirements
            if len(filtered_timestamps) < 50:
                raise ValueError(
                    f"Insufficient activity data: {len(filtered_timestamps)} "
                    f"< 50 minimum points"
                )
            
            # Quality assessment
            quality_metrics = self._assess_activity_data_quality(
                filtered_timestamps, filtered_activity_levels, window_start, window_end
            )
            
            # Calculate hourly activity distributions
            hourly_distributions = self._calculate_hourly_activity_distributions(
                filtered_timestamps, filtered_activity_levels, window_start, window_end
            )
            
            # Calculate fragmentation entropy
            fragmentation_metrics = self._calculate_activity_fragmentation(
                hourly_distributions
            )
            
            # Prepare results
            result = {
                'activity_fragmentation': fragmentation_metrics,
                'mean_entropy': fragmentation_metrics['mean_entropy'],
                'daily_entropies': fragmentation_metrics['daily_entropies'],
                'hourly_patterns': fragmentation_metrics['hourly_patterns'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.fragmentation_config.analysis_window_days,
                    'time_resolution_minutes': self.fragmentation_config.time_resolution_minutes,
                    'activity_threshold': self.fragmentation_config.activity_threshold,
                    'entropy_base': self.fragmentation_config.entropy_base,
                    'normalize_by_total_activity': self.fragmentation_config.normalize_by_total_activity
                },
                'data_summary': {
                    'total_activity_points': len(filtered_timestamps),
                    'days_analyzed': len(hourly_distributions),
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'mean_entropy': fragmentation_metrics['mean_entropy'],
                        'days_analyzed': len(hourly_distributions),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="activity_fragmentation",
                    construct="routine_stability",
                    input_data_summary={
                        'activity_points': len(filtered_timestamps),
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400,
                        'days_analyzed': len(hourly_distributions)
                    },
                    computation_parameters={
                        'time_resolution_minutes': self.fragmentation_config.time_resolution_minutes,
                        'activity_threshold': self.fragmentation_config.activity_threshold,
                        'entropy_base': self.fragmentation_config.entropy_base
                    },
                    result_summary={
                        'mean_entropy': fragmentation_metrics['mean_entropy'],
                        'entropy_stability': fragmentation_metrics['entropy_stability'],
                        'peak_activity_hour': fragmentation_metrics['peak_activity_hour']
                    },
                    data_quality_metrics=quality_metrics,
                    algorithm_version="1.0.0"
                )
            
            return result
            
        except Exception as e:
            # Record failed operation
            if self.provenance_tracker and operation_id:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': False,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    duration_seconds=0.0
                )
            raise
    
    def circadian_midpoint(self,
                          screen_data: Dict[str, Any],
                          window_start: Optional[datetime] = None,
                          window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate circadian midpoint from sleep/wake patterns.
        
        Feature Name: Circadian Midpoint
        Construct: Routine Stability (RS)
        Mathematical Definition: Midpoint between inferred sleep onset and wake time
        Formal Equation: CM = (sleep_onset + wake_time) / 2 for each day
        Assumptions: Sleep/wake patterns reflect circadian rhythm phase
        Limitations: Sensitive to screen usage patterns and irregular schedules
        Edge Cases: No clear sleep/wake transitions, fragmented sleep, shifted patterns
        Output Schema: Circadian midpoints with phase analysis and quality metrics
        
        Args:
            screen_data: Dictionary with 'timestamp', 'screen_state' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing circadian midpoint values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_circadian_midpoint",
                input_parameters={
                    "analysis_window_days": self.circadian_config.analysis_window_days,
                    "sleep_detection_method": self.circadian_config.sleep_detection_method,
                    "wake_detection_method": self.circadian_config.wake_detection_method,
                    "calculate_phase_shift": self.circadian_config.calculate_phase_shift,
                    "weekend_analysis": self.circadian_config.weekend_analysis
                }
            )
        
        try:
            # Validate input data
            self._validate_screen_data(screen_data)
            
            # Extract and preprocess screen data
            timestamps = self._ensure_datetime(screen_data['timestamp'])
            screen_states = screen_data['screen_state']
            
            # Set analysis window
            if window_start is None:
                window_start = min(timestamps)
            if window_end is None:
                window_end = max(timestamps)
            
            # Filter data to analysis window
            filtered_data = self._filter_screen_data_by_window(
                timestamps, screen_states, window_start, window_end
            )
            filtered_timestamps, filtered_screen_states = filtered_data
            
            # Check minimum data requirements
            if len(filtered_timestamps) < 100:
                raise ValueError(
                    f"Insufficient screen data: {len(filtered_timestamps)} "
                    f"< 100 minimum points"
                )
            
            # Quality assessment
            quality_metrics = self._assess_sleep_data_quality(
                filtered_timestamps, filtered_screen_states, window_start, window_end
            )
            
            # Detect sleep/wake times
            sleep_wake_times = self._detect_sleep_wake_times(
                filtered_timestamps, filtered_screen_states
            )
            
            # Calculate circadian midpoints
            midpoint_metrics = self._calculate_circadian_midpoints(
                sleep_wake_times
            )
            
            # Prepare results
            result = {
                'circadian_midpoint': midpoint_metrics,
                'mean_midpoint_hour': midpoint_metrics['mean_midpoint_hour'],
                'daily_midpoints': midpoint_metrics['daily_midpoints'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.circadian_config.analysis_window_days,
                    'sleep_detection_method': self.circadian_config.sleep_detection_method,
                    'wake_detection_method': self.circadian_config.wake_detection_method,
                    'calculate_phase_shift': self.circadian_config.calculate_phase_shift,
                    'weekend_analysis': self.circadian_config.weekend_analysis
                },
                'data_summary': {
                    'total_screen_points': len(filtered_timestamps),
                    'nights_analyzed': len(sleep_wake_times),
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'mean_midpoint_hour': midpoint_metrics['mean_midpoint_hour'],
                        'nights_analyzed': len(sleep_wake_times),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="circadian_midpoint",
                    construct="routine_stability",
                    input_data_summary={
                        'screen_points': len(filtered_timestamps),
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400,
                        'sleep_wake_pairs': len(sleep_wake_times)
                    },
                    computation_parameters={
                        'sleep_detection_method': self.circadian_config.sleep_detection_method,
                        'wake_detection_method': self.circadian_config.wake_detection_method,
                        'calculate_phase_shift': self.circadian_config.calculate_phase_shift
                    },
                    result_summary={
                        'mean_midpoint_hour': midpoint_metrics['mean_midpoint_hour'],
                        'midpoint_sd_hours': midpoint_metrics['midpoint_sd_hours'],
                        'phase_shift_mean_hours': midpoint_metrics.get('phase_shift_mean_hours', 0)
                    },
                    data_quality_metrics=quality_metrics,
                    algorithm_version="1.0.0"
                )
            
            return result
            
        except Exception as e:
            # Record failed operation
            if self.provenance_tracker and operation_id:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': False,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    duration_seconds=0.0
                )
            raise
    
    # Helper methods
    
    def _validate_screen_data(self, data: Dict[str, Any]) -> None:
        """Validate screen data format and content."""
        required_columns = ['timestamp', 'screen_state']
        
        for col in required_columns:
            if col not in data:
                raise ValueError(f"Missing required column: {col}")
        
        # Check equal lengths
        lengths = [len(data[col]) for col in required_columns]
        if len(set(lengths)) > 1:
            raise ValueError("All screen data columns must have equal length")
        
        if len(data['timestamp']) == 0:
            raise ValueError("Screen data cannot be empty")
        
        # Validate screen state values
        valid_states = {0, 1}  # 0=off, 1=on
        for state in data['screen_state']:
            if state not in valid_states:
                raise ValueError(f"Invalid screen state: {state}")
    
    def _validate_activity_data(self, data: Dict[str, Any]) -> None:
        """Validate activity data format and content."""
        required_columns = ['timestamp', 'activity_level']
        
        for col in required_columns:
            if col not in data:
                raise ValueError(f"Missing required column: {col}")
        
        # Check equal lengths
        lengths = [len(data[col]) for col in required_columns]
        if len(set(lengths)) > 1:
            raise ValueError("All activity data columns must have equal length")
        
        if len(data['timestamp']) == 0:
            raise ValueError("Activity data cannot be empty")
        
        # Validate activity level values
        for level in data['activity_level']:
            if not isinstance(level, (int, float)) or level < 0:
                raise ValueError(f"Invalid activity level: {level}")
    
    def _ensure_datetime(self, timestamps: List[Any]) -> List[datetime]:
        """Convert timestamps to datetime objects."""
        datetime_timestamps = []
        for ts in timestamps:
            if isinstance(ts, datetime):
                datetime_timestamps.append(ts)
            elif isinstance(ts, str):
                try:
                    datetime_timestamps.append(datetime.fromisoformat(ts))
                except ValueError:
                    raise ValueError(f"Invalid timestamp format: {ts}")
            else:
                raise ValueError(f"Unsupported timestamp type: {type(ts)}")
        
        return datetime_timestamps
    
    def _filter_screen_data_by_window(self, 
                                     timestamps: List[datetime], 
                                     screen_states: List[int],
                                     window_start: datetime, 
                                     window_end: datetime) -> Tuple[List[datetime], List[int]]:
        """Filter screen data to specified analysis window."""
        filtered_timestamps = []
        filtered_screen_states = []
        
        for i, ts in enumerate(timestamps):
            if window_start <= ts <= window_end:
                filtered_timestamps.append(ts)
                filtered_screen_states.append(screen_states[i])
        
        return filtered_timestamps, filtered_screen_states
    
    def _filter_activity_data_by_window(self, 
                                       timestamps: List[datetime], 
                                       activity_levels: List[float],
                                       window_start: datetime, 
                                       window_end: datetime) -> Tuple[List[datetime], List[float]]:
        """Filter activity data to specified analysis window."""
        filtered_timestamps = []
        filtered_activity_levels = []
        
        for i, ts in enumerate(timestamps):
            if window_start <= ts <= window_end:
                filtered_timestamps.append(ts)
                filtered_activity_levels.append(activity_levels[i])
        
        return filtered_timestamps, filtered_activity_levels
    
    def _assess_sleep_data_quality(self, 
                                  timestamps: List[datetime], 
                                  screen_states: List[int],
                                  window_start: datetime,
                                  window_end: datetime) -> Dict[str, Any]:
        """Assess quality of sleep-related screen data."""
        if len(timestamps) < 2:
            return {
                'coverage_ratio': 0.0,
                'data_completeness': 0.0,
                'temporal_consistency': 0.0,
                'overall_quality': 0.0
            }
        
        # Calculate coverage ratio
        time_span_days = (window_end - window_start).days
        expected_points = time_span_days * 24 * 60  # Assuming 1 point per minute
        coverage_ratio = min(len(timestamps) / expected_points, 1.0) if expected_points > 0 else 0
        
        # Calculate data completeness (percentage of time with data)
        if len(timestamps) > 1:
            total_time = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
            expected_time = time_span_days * 24
            data_completeness = min(total_time / expected_time, 1.0) if expected_time > 0 else 0
        else:
            data_completeness = 0.0
        
        # Calculate temporal consistency (regular intervals)
        if len(timestamps) > 2:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 
                        for i in range(len(timestamps)-1)]
            interval_std = statistics.stdev(intervals) if len(intervals) > 1 else 0
            interval_mean = statistics.mean(intervals)
            temporal_consistency = 1.0 - (interval_std / interval_mean) if interval_mean > 0 else 0
            temporal_consistency = max(0, temporal_consistency)
        else:
            temporal_consistency = 0.0
        
        # Overall quality score
        quality_score = (
            coverage_ratio * 0.4 +
            data_completeness * 0.3 +
            temporal_consistency * 0.3
        )
        
        return {
            'coverage_ratio': coverage_ratio,
            'data_completeness': data_completeness,
            'temporal_consistency': temporal_consistency,
            'overall_quality': quality_score
        }
    
    def _assess_activity_data_quality(self, 
                                     timestamps: List[datetime], 
                                     activity_levels: List[float],
                                     window_start: datetime,
                                     window_end: datetime) -> Dict[str, Any]:
        """Assess quality of activity data."""
        if len(timestamps) < 2:
            return {
                'coverage_ratio': 0.0,
                'data_completeness': 0.0,
                'activity_range': 0.0,
                'overall_quality': 0.0
            }
        
        # Calculate coverage ratio
        time_span_days = (window_end - window_start).days
        expected_points = time_span_days * 24 * 12  # Assuming 5-minute intervals
        coverage_ratio = min(len(timestamps) / expected_points, 1.0) if expected_points > 0 else 0
        
        # Calculate data completeness
        if len(timestamps) > 1:
            total_time = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
            expected_time = time_span_days * 24
            data_completeness = min(total_time / expected_time, 1.0) if expected_time > 0 else 0
        else:
            data_completeness = 0.0
        
        # Calculate activity range
        if activity_levels:
            activity_range = max(activity_levels) - min(activity_levels)
        else:
            activity_range = 0.0
        
        # Overall quality score
        quality_score = (
            coverage_ratio * 0.4 +
            data_completeness * 0.3 +
            min(1.0, activity_range / 10) * 0.3  # Normalize activity range
        )
        
        return {
            'coverage_ratio': coverage_ratio,
            'data_completeness': data_completeness,
            'activity_range': activity_range,
            'overall_quality': quality_score
        }
    
    def _detect_sleep_onset_times(self, 
                                 timestamps: List[datetime], 
                                 screen_states: List[int]) -> List[Dict[str, Any]]:
        """Detect sleep onset times from screen data."""
        sleep_onsets = []
        
        # Group by day
        daily_data = {}
        for ts, state in zip(timestamps, screen_states):
            date_key = ts.date()
            if date_key not in daily_data:
                daily_data[date_key] = []
            daily_data[date_key].append((ts, state))
        
        # Process each day
        for date, day_data in daily_data.items():
            if len(day_data) < 10:  # Minimum data points for reliable detection
                continue
            
            # Sort by timestamp
            day_data.sort()
            
            # Find longest screen-off interval
            longest_interval = None
            longest_duration = 0
            
            for i in range(len(day_data) - 1):
                current_time, current_state = day_data[i]
                next_time, next_state = day_data[i + 1]
                
                # Look for screen-off to screen-on transitions
                if current_state == 0 and next_state == 1:
                    # Find the start of this screen-off period
                    screen_off_start = current_time
                    
                    # Find the end of this screen-off period
                    screen_off_end = next_time
                    
                    duration = (screen_off_end - screen_off_start).total_seconds() / 3600  # hours
                    
                    # Check if this is a valid sleep interval
                    if (self.sleep_onset_config.min_screen_off_duration_hours <= duration <= 
                        self.sleep_onset_config.max_screen_off_duration_hours):
                        
                        if duration > longest_duration:
                            longest_duration = duration
                            longest_interval = (screen_off_start, screen_off_end)
            
            if longest_interval:
                sleep_onset_time = longest_interval[0]
                
                # Convert to hour of day (0-24)
                hour_of_day = sleep_onset_time.hour + sleep_onset_time.minute / 60
                
                sleep_onsets.append({
                    'date': date.isoformat(),
                    'sleep_onset_time': sleep_onset_time.isoformat(),
                    'sleep_onset_hour': hour_of_day,
                    'sleep_duration_hours': longest_duration
                })
        
        return sleep_onsets
    
    def _detect_sleep_intervals(self, 
                               timestamps: List[datetime], 
                               screen_states: List[int]) -> List[Dict[str, Any]]:
        """Detect sleep intervals from screen data."""
        sleep_intervals = []
        
        # Group by day
        daily_data = {}
        for ts, state in zip(timestamps, screen_states):
            date_key = ts.date()
            if date_key not in daily_data:
                daily_data[date_key] = []
            daily_data[date_key].append((ts, state))
        
        # Process each day
        for date, day_data in daily_data.items():
            if len(day_data) < 10:
                continue
            
            # Sort by timestamp
            day_data.sort()
            
            # Find longest screen-off interval
            longest_interval = None
            longest_duration = 0
            
            for i in range(len(day_data) - 1):
                current_time, current_state = day_data[i]
                next_time, next_state = day_data[i + 1]
                
                if current_state == 0 and next_state == 1:
                    screen_off_start = current_time
                    screen_off_end = next_time
                    
                    duration = (screen_off_end - screen_off_start).total_seconds() / 3600
                    
                    if (self.sleep_duration_config.min_sleep_duration_hours <= duration <= 
                        self.sleep_duration_config.max_sleep_duration_hours):
                        
                        if duration > longest_duration:
                            longest_duration = duration
                            longest_interval = (screen_off_start, screen_off_end)
            
            if longest_interval:
                sleep_intervals.append({
                    'date': date.isoformat(),
                    'sleep_start_time': longest_interval[0].isoformat(),
                    'sleep_end_time': longest_interval[1].isoformat(),
                    'sleep_duration_hours': longest_duration
                })
        
        return sleep_intervals
    
    def _calculate_sleep_onset_consistency(self, 
                                         sleep_onsets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sleep onset consistency metrics."""
        if not sleep_onsets:
            return {
                'sleep_onset_sd_hours': 0.0,
                'mean_sleep_onset_hour': 0.0,
                'sleep_onset_range_hours': 0.0,
                'daily_sleep_onsets': []
            }
        
        # Extract sleep onset hours
        onset_hours = [onset['sleep_onset_hour'] for onset in sleep_onsets]
        
        # Calculate statistics
        mean_onset = statistics.mean(onset_hours)
        onset_sd = statistics.stdev(onset_hours) if len(onset_hours) > 1 else 0.0
        onset_range = max(onset_hours) - min(onset_hours)
        
        # Handle circular nature of time (e.g., 23:00 and 01:00)
        if onset_range > 12:  # Likely crossing midnight
            # Adjust times to handle circularity
            adjusted_hours = []
            for hour in onset_hours:
                if hour < 12:
                    adjusted_hours.append(hour + 24)
                else:
                    adjusted_hours.append(hour)
            
            mean_onset = statistics.mean(adjusted_hours)
            onset_sd = statistics.stdev(adjusted_hours) if len(adjusted_hours) > 1 else 0.0
            mean_onset = mean_onset % 24
        
        # Outlier detection if configured
        if self.sleep_onset_config.outlier_detection and len(onset_hours) > 3:
            mean_val = statistics.mean(onset_hours)
            std_val = statistics.stdev(onset_hours)
            
            filtered_onsets = []
            for onset in sleep_onsets:
                if abs(onset['sleep_onset_hour'] - mean_val) <= 2 * std_val:
                    filtered_onsets.append(onset)
            
            if len(filtered_onsets) >= 3:
                sleep_onsets = filtered_onsets
                onset_hours = [onset['sleep_onset_hour'] for onset in sleep_onsets]
                mean_onset = statistics.mean(onset_hours)
                onset_sd = statistics.stdev(onset_hours) if len(onset_hours) > 1 else 0.0
        
        return {
            'sleep_onset_sd_hours': onset_sd,
            'mean_sleep_onset_hour': mean_onset,
            'sleep_onset_range_hours': onset_range,
            'daily_sleep_onsets': sleep_onsets
        }
    
    def _calculate_sleep_duration_metrics(self, 
                                        sleep_intervals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sleep duration metrics."""
        if not sleep_intervals:
            return {
                'mean_sleep_duration_hours': 0.0,
                'sleep_duration_sd_hours': 0.0,
                'sleep_efficiency': 0.0,
                'daily_sleep_durations': []
            }
        
        # Extract sleep durations
        durations = [interval['sleep_duration_hours'] for interval in sleep_intervals]
        
        # Calculate statistics
        mean_duration = statistics.mean(durations)
        duration_sd = statistics.stdev(durations) if len(durations) > 1 else 0.0
        
        # Calculate sleep efficiency (assuming 8 hours as optimal)
        optimal_sleep = 8.0
        sleep_efficiency = 1.0 - abs(mean_duration - optimal_sleep) / optimal_sleep
        sleep_efficiency = max(0, sleep_efficiency)
        
        return {
            'mean_sleep_duration_hours': mean_duration,
            'sleep_duration_sd_hours': duration_sd,
            'sleep_efficiency': sleep_efficiency,
            'daily_sleep_durations': sleep_intervals
        }
    
    def _calculate_hourly_activity_distributions(self, 
                                               timestamps: List[datetime], 
                                               activity_levels: List[float],
                                               window_start: datetime,
                                               window_end: datetime) -> Dict[str, Dict[int, float]]:
        """Calculate hourly activity distributions for each day."""
        daily_distributions = {}
        
        # Group by day
        daily_data = {}
        for ts, activity in zip(timestamps, activity_levels):
            date_key = ts.date()
            if date_key not in daily_data:
                daily_data[date_key] = []
            daily_data[date_key].append((ts, activity))
        
        # Process each day
        for date, day_data in daily_data.items():
            # Initialize hourly bins
            hourly_activity = {hour: 0.0 for hour in range(24)}
            
            # Aggregate activity by hour
            for ts, activity in day_data:
                hour = ts.hour
                hourly_activity[hour] += activity
            
            # Normalize if configured
            if self.fragmentation_config.normalize_by_total_activity:
                total_activity = sum(hourly_activity.values())
                if total_activity > 0:
                    for hour in hourly_activity:
                        hourly_activity[hour] /= total_activity
            
            daily_distributions[date.isoformat()] = hourly_activity
        
        return daily_distributions
    
    def _calculate_activity_fragmentation(self, 
                                        hourly_distributions: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
        """Calculate activity fragmentation entropy."""
        if not hourly_distributions:
            return {
                'mean_entropy': 0.0,
                'entropy_stability': 0.0,
                'peak_activity_hour': 12,
                'daily_entropies': {},
                'hourly_patterns': {}
            }
        
        daily_entropies = {}
        hourly_patterns = {hour: [] for hour in range(24)}
        
        # Calculate entropy for each day
        for date, hourly_activity in hourly_distributions.items():
            # Calculate Shannon entropy
            entropy = 0.0
            total_activity = sum(hourly_activity.values())
            
            if total_activity > 0:
                for hour, activity in hourly_activity.items():
                    if activity > 0:
                        probability = activity / total_activity
                        if self.fragmentation_config.entropy_base == "natural":
                            entropy -= probability * math.log(probability)
                        else:  # log2
                            entropy -= probability * math.log2(probability)
            
            daily_entropies[date] = entropy
            
            # Store hourly patterns for averaging
            for hour, activity in hourly_activity.items():
                hourly_patterns[hour].append(activity)
        
        # Calculate mean entropy and stability
        entropy_values = list(daily_entropies.values())
        mean_entropy = statistics.mean(entropy_values) if entropy_values else 0.0
        entropy_sd = statistics.stdev(entropy_values) if len(entropy_values) > 1 else 0.0
        entropy_stability = 1.0 - (entropy_sd / mean_entropy) if mean_entropy > 0 else 0.0
        
        # Find peak activity hour
        mean_hourly_activity = {}
        for hour, values in hourly_patterns.items():
            mean_hourly_activity[hour] = statistics.mean(values) if values else 0.0
        
        peak_activity_hour = max(mean_hourly_activity, key=mean_hourly_activity.get)
        
        return {
            'mean_entropy': mean_entropy,
            'entropy_stability': entropy_stability,
            'peak_activity_hour': peak_activity_hour,
            'daily_entropies': daily_entropies,
            'hourly_patterns': mean_hourly_activity
        }
    
    def _detect_sleep_wake_times(self, 
                                timestamps: List[datetime], 
                                screen_states: List[int]) -> List[Dict[str, Any]]:
        """Detect sleep and wake times from screen data."""
        sleep_wake_times = []
        
        # Group by day
        daily_data = {}
        for ts, state in zip(timestamps, screen_states):
            date_key = ts.date()
            if date_key not in daily_data:
                daily_data[date_key] = []
            daily_data[date_key].append((ts, state))
        
        # Process each day
        for date, day_data in daily_data.items():
            if len(day_data) < 10:
                continue
            
            # Sort by timestamp
            day_data.sort()
            
            # Find sleep and wake times
            sleep_time = None
            wake_time = None
            
            # Find sleep time (last screen-off before longest sleep period)
            longest_sleep = None
            longest_duration = 0
            
            for i in range(len(day_data) - 1):
                current_time, current_state = day_data[i]
                next_time, next_state = day_data[i + 1]
                
                if current_state == 0 and next_state == 1:
                    duration = (next_time - current_time).total_seconds() / 3600
                    
                    if (self.circadian_config.sleep_detection_method == "screen_off" and
                        2 <= duration <= 12):
                        
                        if duration > longest_duration:
                            longest_duration = duration
                            longest_sleep = (current_time, next_time)
            
            if longest_sleep:
                sleep_time = longest_sleep[0]
                wake_time = longest_sleep[1]
                
                # Convert to hours of day
                sleep_hour = sleep_time.hour + sleep_time.minute / 60
                wake_hour = wake_time.hour + wake_time.minute / 60
                
                # Calculate circadian midpoint
                if wake_hour > sleep_hour:
                    midpoint = (sleep_hour + wake_hour) / 2
                else:  # Crosses midnight
                    midpoint = ((sleep_hour + 24) + wake_hour) / 2
                    if midpoint >= 24:
                        midpoint -= 24
                
                sleep_wake_times.append({
                    'date': date.isoformat(),
                    'sleep_time': sleep_time.isoformat(),
                    'wake_time': wake_time.isoformat(),
                    'sleep_hour': sleep_hour,
                    'wake_hour': wake_hour,
                    'midpoint_hour': midpoint
                })
        
        return sleep_wake_times
    
    def _calculate_circadian_midpoints(self, 
                                     sleep_wake_times: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate circadian midpoint metrics."""
        if not sleep_wake_times:
            return {
                'mean_midpoint_hour': 0.0,
                'midpoint_sd_hours': 0.0,
                'phase_shift_mean_hours': 0.0,
                'daily_midpoints': []
            }
        
        # Extract midpoint hours
        midpoint_hours = [sw['midpoint_hour'] for sw in sleep_wake_times]
        
        # Calculate statistics
        mean_midpoint = statistics.mean(midpoint_hours)
        midpoint_sd = statistics.stdev(midpoint_hours) if len(midpoint_hours) > 1 else 0.0
        
        # Calculate phase shifts (day-to-day changes)
        phase_shifts = []
        for i in range(1, len(midpoint_hours)):
            shift = midpoint_hours[i] - midpoint_hours[i-1]
            # Handle circular nature
            if shift > 12:
                shift -= 24
            elif shift < -12:
                shift += 24
            phase_shifts.append(shift)
        
        phase_shift_mean = statistics.mean(phase_shifts) if phase_shifts else 0.0
        phase_shift_sd = statistics.stdev(phase_shifts) if len(phase_shifts) > 1 else 0.0
        
        # Outlier detection if configured
        if self.circadian_config.outlier_detection and len(midpoint_hours) > 3:
            mean_val = statistics.mean(midpoint_hours)
            std_val = statistics.stdev(midpoint_hours)
            
            filtered_midpoints = []
            for sw in sleep_wake_times:
                if abs(sw['midpoint_hour'] - mean_val) <= 2 * std_val:
                    filtered_midpoints.append(sw)
            
            if len(filtered_midpoints) >= 3:
                sleep_wake_times = filtered_midpoints
                midpoint_hours = [sw['midpoint_hour'] for sw in sleep_wake_times]
                mean_midpoint = statistics.mean(midpoint_hours)
                midpoint_sd = statistics.stdev(midpoint_hours) if len(midpoint_hours) > 1 else 0.0
        
        return {
            'mean_midpoint_hour': mean_midpoint,
            'midpoint_sd_hours': midpoint_sd,
            'phase_shift_mean_hours': phase_shift_mean,
            'phase_shift_sd_hours': phase_shift_sd,
            'daily_midpoints': sleep_wake_times
        }
