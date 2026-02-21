"""
Behavioral Activation features for psyconstruct package.

This module implements features related to behavioral activation construct,
including activity volume, location diversity, app usage breadth, and
activity timing variance.

Product: Construct-Aligned Digital Phenotyping Toolkit
Construct: Behavioral Activation (BA)
Measurement Model: Reflective
"""

import math
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings


@dataclass
class ActivityVolumeConfig:
    """Configuration for activity volume feature extraction."""
    
    # Time window for aggregation
    window_hours: int = 24
    
    # Minimum data requirements
    min_data_coverage: float = 0.7  # 70% minimum coverage
    min_sampling_rate_hz: float = 0.1  # Minimum sampling rate
    
    # Quality thresholds
    max_gap_minutes: float = 60.0  # Maximum gap for interpolation
    outlier_threshold_std: float = 3.0  # Outlier detection threshold
    
    # Processing options
    interpolate_gaps: bool = True
    remove_outliers: bool = True


@dataclass
class ActivityTimingVarianceConfig:
    """Configuration for activity timing variance feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 7  # Weekly analysis window
    min_activity_threshold: float = 0.1  # Minimum activity to count as active period
    
    # Time resolution
    time_resolution_minutes: int = 60  # 1-hour bins for timing analysis
    min_active_hours_per_day: int = 4  # Minimum active hours to include day
    
    # Variance calculation parameters
    variance_metric: str = "std"  # "std", "cv", "iqr" - standard deviation, coefficient of variation, interquartile range
    include_weekend_analysis: bool = True  # Separate weekday/weekend analysis
    normalize_by_activity_level: bool = True  # Normalize variance by overall activity
    
    # Data quality requirements
    min_days_with_data: int = 5  # Minimum days with activity data
    min_data_coverage: float = 0.6  # 60% minimum coverage per day
    
    # Processing options
    smooth_activity_data: bool = True  # Apply smoothing to activity data
    outlier_detection: bool = True  # Detect and handle outliers


@dataclass
class AppUsageBreadthConfig:
    """Configuration for app usage breadth feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 7  # Weekly analysis window
    min_usage_duration_seconds: float = 30.0  # Minimum usage to count as meaningful
    min_app_sessions: int = 3  # Minimum sessions to count app as used
    
    # App categorization
    categorize_apps: bool = True  # Group apps by category
    exclude_system_apps: bool = True  # Exclude system apps from diversity
    
    # Entropy calculation parameters
    entropy_base: float = 2.0  # Base for entropy calculation (2 = bits)
    include_duration_weighting: bool = True  # Weight by usage duration
    
    # Data quality requirements
    min_total_sessions: int = 50  # Minimum total app sessions
    min_active_days: int = 3  # Minimum days with app usage
    
    # Processing options
    normalize_by_total_time: bool = False  # Normalize by total usage time


@dataclass
class LocationDiversityConfig:
    """Configuration for location diversity feature extraction."""
    
    # Clustering parameters
    clustering_radius_meters: float = 50.0  # Radius for location clustering
    min_cluster_size: int = 5  # Minimum points to form a cluster
    
    # Time window for analysis
    analysis_window_days: int = 7  # Weekly analysis
    
    # Data quality requirements
    min_gps_points: int = 100  # Minimum GPS points for analysis
    min_accuracy_meters: float = 100.0  # Maximum GPS accuracy for inclusion
    
    # Entropy calculation parameters
    entropy_base: float = 2.0  # Base for entropy calculation (2 = bits)
    min_location_visits: int = 3  # Minimum visits to count location
    
    # Processing options
    remove_home_location: bool = True  # Exclude home cluster from diversity
    accuracy_weighting: bool = True  # Weight by GPS accuracy


class BehavioralActivationFeatures:
    """
    Implementation of Behavioral Activation (BA) construct features.
    
    This class provides methods for extracting features related to behavioral
    activation, which reflects the tendency to engage in goal-directed activities
    and environmental interactions.
    
    Attributes:
        activity_config: Configuration for activity volume features
        location_config: Configuration for location diversity features
        app_usage_config: Configuration for app usage breadth features
        timing_config: Configuration for activity timing variance features
        provenance_tracker: Provenance tracking instance
    """
    
    def __init__(self, 
                 activity_config: Optional[ActivityVolumeConfig] = None,
                 location_config: Optional[LocationDiversityConfig] = None,
                 app_usage_config: Optional[AppUsageBreadthConfig] = None,
                 timing_config: Optional[ActivityTimingVarianceConfig] = None):
        """
        Initialize behavioral activation features extractor.
        
        Args:
            activity_config: Configuration for activity volume features
            location_config: Configuration for location diversity features
            app_usage_config: Configuration for app usage breadth features
            timing_config: Configuration for activity timing variance features
        """
        self.activity_config = activity_config or ActivityVolumeConfig()
        self.location_config = location_config or LocationDiversityConfig()
        self.app_usage_config = app_usage_config or AppUsageBreadthConfig()
        self.timing_config = timing_config or ActivityTimingVarianceConfig()
        
        # Import provenance tracker locally to avoid circular imports
        try:
            from ..utils.provenance import get_provenance_tracker
            self.provenance_tracker = get_provenance_tracker()
        except ImportError:
            self.provenance_tracker = None
    
    def activity_volume(self, 
                       accelerometer_data: Dict[str, Any],
                       window_start: Optional[datetime] = None,
                       window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate activity volume from accelerometer magnitude.
        
        Feature Name: Activity Volume
        Construct: Behavioral Activation (BA)
        Mathematical Definition: Rolling sum of accelerometer magnitude over 24-hour window
        Formal Equation: AV = Σ_{t ∈ 24h} magnitude_t
        Assumptions: Accelerometer magnitude is computed as √(x² + y² + z²) and represents movement intensity
        Limitations: Sensitive to device placement and sampling frequency variations
        Edge Cases: Missing data intervals, device frequency variability, magnitude outliers
        Output Schema: Daily activity volume with timestamps and quality metrics
        
        Args:
            accelerometer_data: Dictionary with 'timestamp', 'x', 'y', 'z' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing activity volume values and metadata
            
        Raises:
            ValueError: If required data columns are missing or invalid
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_activity_volume",
                input_parameters={
                    "window_hours": self.activity_config.window_hours,
                    "min_coverage": self.activity_config.min_data_coverage,
                    "interpolate_gaps": self.activity_config.interpolate_gaps
                }
            )
        
        try:
            # Validate input data
            self._validate_accelerometer_data(accelerometer_data)
            
            # Extract timestamps and compute magnitude
            timestamps = self._ensure_datetime(accelerometer_data['timestamp'])
            magnitude = self._compute_magnitude(
                accelerometer_data['x'],
                accelerometer_data['y'], 
                accelerometer_data['z']
            )
            
            # Set analysis window
            if window_start is None:
                window_start = min(timestamps)
            if window_end is None:
                window_end = max(timestamps)
            
            # Filter data to analysis window
            filtered_data = self._filter_by_window(
                timestamps, magnitude, window_start, window_end
            )
            filtered_timestamps, filtered_magnitude = filtered_data
            
            # Quality assessment
            quality_metrics = self._assess_data_quality(
                filtered_timestamps, filtered_magnitude
            )
            
            # Check if data meets minimum requirements
            if quality_metrics['coverage_ratio'] < self.activity_config.min_data_coverage:
                raise ValueError(
                    f"Insufficient data coverage: {quality_metrics['coverage_ratio']:.3f} "
                    f"< {self.activity_config.min_data_coverage}"
                )
            
            # Preprocess data (interpolation and outlier removal)
            processed_magnitude = self._preprocess_data(
                filtered_timestamps, filtered_magnitude, quality_metrics
            )
            
            # Calculate activity volume for each 24-hour window
            activity_volumes = self._calculate_daily_volumes(
                filtered_timestamps, processed_magnitude, window_start, window_end
            )
            
            # Prepare results
            result = {
                'activity_volume': activity_volumes,
                'timestamps': [av['timestamp'] for av in activity_volumes],
                'values': [av['volume'] for av in activity_volumes],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'window_hours': self.activity_config.window_hours,
                    'min_coverage': self.activity_config.min_data_coverage,
                    'interpolation_applied': self.activity_config.interpolate_gaps,
                    'outlier_removal_applied': self.activity_config.remove_outliers
                },
                'data_summary': {
                    'total_records': len(filtered_timestamps),
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
                        'daily_volumes_calculated': len(activity_volumes),
                        'data_coverage': quality_metrics['coverage_ratio'],
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0  # Would be measured in real implementation
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="activity_volume",
                    construct="behavioral_activation",
                    input_data_summary={
                        'records': len(accelerometer_data['timestamp']),
                        'time_span_hours': (max(timestamps) - min(timestamps)).total_seconds() / 3600,
                        'sampling_rate_hz': quality_metrics['sampling_rate_hz']
                    },
                    computation_parameters={
                        'window_hours': self.activity_config.window_hours,
                        'min_coverage': self.activity_config.min_data_coverage,
                        'interpolate_gaps': self.activity_config.interpolate_gaps,
                        'remove_outliers': self.activity_config.remove_outliers
                    },
                    result_summary={
                        'daily_volumes': len(activity_volumes),
                        'mean_volume': sum(av['volume'] for av in activity_volumes) / len(activity_volumes) if activity_volumes else 0,
                        'volume_range': {
                            'min': min(av['volume'] for av in activity_volumes) if activity_volumes else 0,
                            'max': max(av['volume'] for av in activity_volumes) if activity_volumes else 0
                        }
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
    
    def location_diversity(self,
                          gps_data: Dict[str, Any],
                          window_start: Optional[datetime] = None,
                          window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate location diversity from GPS coordinates.
        
        Feature Name: Location Diversity
        Construct: Behavioral Activation (BA)
        Mathematical Definition: Shannon entropy of clustered GPS locations per week
        Formal Equation: LD = -Σ_{i=1}^{n} p_i * log_b(p_i) where p_i = visits_i / total_visits
        Assumptions: GPS coordinates are accurate and represent meaningful location visits
        Limitations: Sensitive to GPS accuracy, clustering radius, and sampling frequency
        Edge Cases: Sparse GPS data, GPS accuracy issues, single location patterns
        Output Schema: Weekly location diversity with entropy values and cluster information
        
        Args:
            gps_data: Dictionary with 'timestamp', 'latitude', 'longitude' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing location diversity values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_location_diversity",
                input_parameters={
                    "clustering_radius_meters": self.location_config.clustering_radius_meters,
                    "min_cluster_size": self.location_config.min_cluster_size,
                    "analysis_window_days": self.location_config.analysis_window_days,
                    "entropy_base": self.location_config.entropy_base
                }
            )
        
        try:
            # Validate input data
            self._validate_gps_data(gps_data)
            
            # Extract and preprocess GPS data
            timestamps = self._ensure_datetime(gps_data['timestamp'])
            latitudes = gps_data['latitude']
            longitudes = gps_data['longitude']
            
            # Set analysis window
            if window_start is None:
                window_start = min(timestamps)
            if window_end is None:
                window_end = max(timestamps)
            
            # Filter data to analysis window
            filtered_data = self._filter_gps_by_window(
                timestamps, latitudes, longitudes, window_start, window_end
            )
            filtered_timestamps, filtered_latitudes, filtered_longitudes = filtered_data
            
            # Check minimum data requirements
            if len(filtered_timestamps) < self.location_config.min_gps_points:
                raise ValueError(
                    f"Insufficient GPS points: {len(filtered_timestamps)} "
                    f"< {self.location_config.min_gps_points}"
                )
            
            # Quality assessment
            quality_metrics = self._assess_gps_quality(
                filtered_timestamps, filtered_latitudes, filtered_longitudes
            )
            
            # Perform location clustering
            clusters = self._perform_location_clustering(
                filtered_latitudes, filtered_longitudes, quality_metrics
            )
            
            # Calculate location diversity metrics
            diversity_metrics = self._calculate_location_diversity(
                clusters, filtered_timestamps
            )
            
            # Prepare results
            result = {
                'location_diversity': diversity_metrics,
                'weekly_entropy': diversity_metrics['weekly_entropy'],
                'cluster_count': diversity_metrics['cluster_count'],
                'unique_locations': diversity_metrics['unique_locations'],
                'clusters': clusters,
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'clustering_radius_meters': self.location_config.clustering_radius_meters,
                    'min_cluster_size': self.location_config.min_cluster_size,
                    'analysis_window_days': self.location_config.analysis_window_days,
                    'entropy_base': self.location_config.entropy_base,
                    'remove_home_location': self.location_config.remove_home_location
                },
                'data_summary': {
                    'total_gps_points': len(filtered_timestamps),
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
                        'clusters_found': len(clusters),
                        'weekly_entropy': diversity_metrics['weekly_entropy'],
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="location_diversity",
                    construct="behavioral_activation",
                    input_data_summary={
                        'gps_points': len(filtered_timestamps),
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400,
                        'geographic_area': quality_metrics['geographic_area']
                    },
                    computation_parameters={
                        'clustering_radius_meters': self.location_config.clustering_radius_meters,
                        'min_cluster_size': self.location_config.min_cluster_size,
                        'entropy_base': self.location_config.entropy_base,
                        'remove_home_location': self.location_config.remove_home_location
                    },
                    result_summary={
                        'weekly_entropy': diversity_metrics['weekly_entropy'],
                        'cluster_count': diversity_metrics['cluster_count'],
                        'unique_locations': diversity_metrics['unique_locations'],
                        'home_cluster_removed': diversity_metrics.get('home_cluster_removed', False)
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
    
    def app_usage_breadth(self,
                         app_usage_data: Dict[str, Any],
                         window_start: Optional[datetime] = None,
                         window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate app usage breadth from app usage logs.
        
        Feature Name: App Usage Breadth
        Construct: Behavioral Activation (BA)
        Mathematical Definition: Shannon entropy of app usage patterns per week
        Formal Equation: AUB = -Σ_{i=1}^{n} p_i * log_b(p_i) where p_i = usage_time_i / total_usage_time
        Assumptions: App usage data represents meaningful engagement and is accurately recorded
        Limitations: Sensitive to app categorization, session detection, and usage duration accuracy
        Edge Cases: Sparse usage data, single-app dominance, system app inclusion
        Output Schema: Weekly app usage breadth with entropy values and app usage patterns
        
        Args:
            app_usage_data: Dictionary with 'timestamp', 'app_name', 'duration_seconds' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing app usage breadth values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_app_usage_breadth",
                input_parameters={
                    "analysis_window_days": self.app_usage_config.analysis_window_days,
                    "min_usage_duration": self.app_usage_config.min_usage_duration_seconds,
                    "min_app_sessions": self.app_usage_config.min_app_sessions,
                    "entropy_base": self.app_usage_config.entropy_base,
                    "categorize_apps": self.app_usage_config.categorize_apps
                }
            )
        
        try:
            # Validate input data
            self._validate_app_usage_data(app_usage_data)
            
            # Extract and preprocess app usage data
            timestamps = self._ensure_datetime(app_usage_data['timestamp'])
            app_names = app_usage_data['app_name']
            durations = app_usage_data['duration_seconds']
            
            # Set analysis window
            if window_start is None:
                window_start = min(timestamps)
            if window_end is None:
                window_end = max(timestamps)
            
            # Filter data to analysis window
            filtered_data = self._filter_app_usage_by_window(
                timestamps, app_names, durations, window_start, window_end
            )
            filtered_timestamps, filtered_app_names, filtered_durations = filtered_data
            
            # Check minimum data requirements
            if len(filtered_timestamps) < self.app_usage_config.min_total_sessions:
                raise ValueError(
                    f"Insufficient app usage sessions: {len(filtered_timestamps)} "
                    f"< {self.app_usage_config.min_total_sessions}"
                )
            
            # Quality assessment
            quality_metrics = self._assess_app_usage_quality(
                filtered_timestamps, filtered_app_names, filtered_durations
            )
            
            # Process app usage data
            processed_usage = self._process_app_usage_data(
                filtered_timestamps, filtered_app_names, filtered_durations
            )
            
            # Calculate app usage breadth metrics
            breadth_metrics = self._calculate_app_usage_breadth(
                processed_usage, filtered_timestamps
            )
            
            # Prepare results
            result = {
                'app_usage_breadth': breadth_metrics,
                'weekly_entropy': breadth_metrics['weekly_entropy'],
                'unique_apps': breadth_metrics['unique_apps'],
                'total_sessions': breadth_metrics['total_sessions'],
                'total_usage_time': breadth_metrics['total_usage_time'],
                'app_usage_patterns': breadth_metrics['app_usage_patterns'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.app_usage_config.analysis_window_days,
                    'min_usage_duration': self.app_usage_config.min_usage_duration_seconds,
                    'min_app_sessions': self.app_usage_config.min_app_sessions,
                    'entropy_base': self.app_usage_config.entropy_base,
                    'categorize_apps': self.app_usage_config.categorize_apps,
                    'exclude_system_apps': self.app_usage_config.exclude_system_apps,
                    'include_duration_weighting': self.app_usage_config.include_duration_weighting
                },
                'data_summary': {
                    'total_sessions': len(filtered_timestamps),
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
                        'unique_apps_found': breadth_metrics['unique_apps'],
                        'weekly_entropy': breadth_metrics['weekly_entropy'],
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="app_usage_breadth",
                    construct="behavioral_activation",
                    input_data_summary={
                        'app_sessions': len(filtered_timestamps),
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400,
                        'unique_apps_raw': len(set(filtered_app_names))
                    },
                    computation_parameters={
                        'analysis_window_days': self.app_usage_config.analysis_window_days,
                        'min_usage_duration': self.app_usage_config.min_usage_duration_seconds,
                        'min_app_sessions': self.app_usage_config.min_app_sessions,
                        'entropy_base': self.app_usage_config.entropy_base,
                        'categorize_apps': self.app_usage_config.categorize_apps
                    },
                    result_summary={
                        'weekly_entropy': breadth_metrics['weekly_entropy'],
                        'unique_apps': breadth_metrics['unique_apps'],
                        'total_sessions': breadth_metrics['total_sessions'],
                        'total_usage_time': breadth_metrics['total_usage_time'],
                        'dominant_app': breadth_metrics.get('dominant_app', 'N/A')
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
    
    def activity_timing_variance(self,
                                accelerometer_data: Dict[str, Any],
                                window_start: Optional[datetime] = None,
                                window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate activity timing variance from accelerometer data.
        
        Feature Name: Activity Timing Variance
        Construct: Behavioral Activation (BA)
        Mathematical Definition: Variance of activity timing patterns across days
        Formal Equation: ATV = Var(activity_timing_distribution) where timing is measured in hourly bins
        Assumptions: Accelerometer data represents meaningful activity and timing patterns are stable
        Limitations: Sensitive to missing data, device wear patterns, and activity threshold selection
        Edge Cases: Sparse activity data, single-day patterns, irregular sleep schedules
        Output Schema: Weekly timing variance with hourly patterns and variance metrics
        
        Args:
            accelerometer_data: Dictionary with 'timestamp', 'x', 'y', 'z' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing activity timing variance values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_activity_timing_variance",
                input_parameters={
                    "analysis_window_days": self.timing_config.analysis_window_days,
                    "time_resolution_minutes": self.timing_config.time_resolution_minutes,
                    "variance_metric": self.timing_config.variance_metric,
                    "min_activity_threshold": self.timing_config.min_activity_threshold,
                    "include_weekend_analysis": self.timing_config.include_weekend_analysis
                }
            )
        
        try:
            # Validate input data
            self._validate_accelerometer_data(accelerometer_data)
            
            # Extract and preprocess accelerometer data
            timestamps = self._ensure_datetime(accelerometer_data['timestamp'])
            magnitude = self._compute_magnitude(
                accelerometer_data['x'],
                accelerometer_data['y'], 
                accelerometer_data['z']
            )
            
            # Set analysis window
            if window_start is None:
                window_start = min(timestamps)
            if window_end is None:
                window_end = max(timestamps)
            
            # Filter data to analysis window
            filtered_data = self._filter_by_window(
                timestamps, magnitude, window_start, window_end
            )
            filtered_timestamps, filtered_magnitude = filtered_data
            
            # Check minimum data requirements
            if len(filtered_timestamps) < 100:  # Minimum data points
                raise ValueError(
                    f"Insufficient accelerometer data: {len(filtered_timestamps)} "
                    f"< 100 minimum points"
                )
            
            # Quality assessment
            quality_metrics = self._assess_timing_data_quality(
                filtered_timestamps, filtered_magnitude
            )
            
            # Process activity timing data
            timing_patterns = self._process_activity_timing(
                filtered_timestamps, filtered_magnitude, window_start, window_end
            )
            
            # Calculate timing variance metrics
            variance_metrics = self._calculate_timing_variance(
                timing_patterns, filtered_timestamps
            )
            
            # Prepare results
            result = {
                'activity_timing_variance': variance_metrics,
                'weekly_variance': variance_metrics['weekly_variance'],
                'hourly_patterns': variance_metrics['hourly_patterns'],
                'daily_variances': variance_metrics['daily_variances'],
                'variance_by_day_type': variance_metrics['variance_by_day_type'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.timing_config.analysis_window_days,
                    'time_resolution_minutes': self.timing_config.time_resolution_minutes,
                    'variance_metric': self.timing_config.variance_metric,
                    'min_activity_threshold': self.timing_config.min_activity_threshold,
                    'include_weekend_analysis': self.timing_config.include_weekend_analysis,
                    'normalize_by_activity_level': self.timing_config.normalize_by_activity_level,
                    'min_active_hours_per_day': self.timing_config.min_active_hours_per_day
                },
                'data_summary': {
                    'total_records': len(filtered_timestamps),
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days,
                    'days_analyzed': len(timing_patterns['daily_patterns'])
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'weekly_variance': variance_metrics['weekly_variance'],
                        'days_analyzed': len(timing_patterns['daily_patterns']),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="activity_timing_variance",
                    construct="behavioral_activation",
                    input_data_summary={
                        'accelerometer_points': len(filtered_timestamps),
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400,
                        'sampling_rate_hz': quality_metrics['sampling_rate_hz']
                    },
                    computation_parameters={
                        'analysis_window_days': self.timing_config.analysis_window_days,
                        'time_resolution_minutes': self.timing_config.time_resolution_minutes,
                        'variance_metric': self.timing_config.variance_metric,
                        'min_activity_threshold': self.timing_config.min_activity_threshold
                    },
                    result_summary={
                        'weekly_variance': variance_metrics['weekly_variance'],
                        'peak_activity_hour': variance_metrics.get('peak_activity_hour', 'N/A'),
                        'variance_stability': variance_metrics.get('variance_stability', 'N/A'),
                        'weekday_weekend_diff': variance_metrics.get('weekday_weekend_difference', 'N/A')
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
    
    # Helper methods for activity timing variance
    
    def _assess_timing_data_quality(self, 
                                   timestamps: List[datetime], 
                                   magnitude: List[float]) -> Dict[str, Any]:
        """Assess quality of timing data for variance analysis."""
        if len(timestamps) < 2:
            return {
                'coverage_ratio': 0.0,
                'sampling_rate_hz': 0.0,
                'data_completeness': 0.0,
                'temporal_consistency': 0.0,
                'overall_quality': 0.0
            }
        
        # Calculate sampling rate
        time_span = (max(timestamps) - min(timestamps)).total_seconds()
        sampling_rate_hz = len(timestamps) / time_span if time_span > 0 else 0
        
        # Calculate data coverage (expected vs actual samples)
        expected_samples = time_span * sampling_rate_hz
        coverage_ratio = min(len(timestamps) / expected_samples, 1.0) if expected_samples > 0 else 0
        
        # Calculate temporal consistency (regularity of sampling)
        if len(timestamps) > 1:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            mean_interval = sum(intervals) / len(intervals)
            variance_interval = sum((x - mean_interval)**2 for x in intervals) / len(intervals)
            std_interval = math.sqrt(variance_interval)
            # Higher consistency = lower coefficient of variation
            temporal_consistency = 1.0 - (std_interval / mean_interval) if mean_interval > 0 else 0
            temporal_consistency = max(0, min(1, temporal_consistency))
        else:
            temporal_consistency = 0
        
        # Calculate data completeness (24-hour coverage)
        hours_with_data = set(ts.hour for ts in timestamps)
        data_completeness = len(hours_with_data) / 24.0
        
        # Overall quality score
        quality_score = (
            coverage_ratio * 0.3 +
            min(1.0, sampling_rate_hz / 10) * 0.3 +  # 10 Hz as reference
            data_completeness * 0.2 +
            temporal_consistency * 0.2
        )
        
        return {
            'coverage_ratio': coverage_ratio,
            'sampling_rate_hz': sampling_rate_hz,
            'data_completeness': data_completeness,
            'temporal_consistency': temporal_consistency,
            'overall_quality': quality_score
        }
    
    def _process_activity_timing(self, 
                                timestamps: List[datetime], 
                                magnitude: List[float],
                                window_start: datetime,
                                window_end: datetime) -> Dict[str, Any]:
        """Process activity data into timing patterns."""
        # Create hourly bins
        hours_per_day = 24 * 60 // self.timing_config.time_resolution_minutes
        daily_patterns = {}
        
        # Group data by day
        current_date = window_start.date()
        end_date = window_end.date()
        
        while current_date <= end_date:
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = day_start + timedelta(days=1)
            
            # Initialize hourly bins for this day
            hourly_activity = [0.0] * hours_per_day
            
            # Process data for this day
            for i, ts in enumerate(timestamps):
                if day_start <= ts < day_end:
                    hour_bin = int((ts - day_start).total_seconds() / (self.timing_config.time_resolution_minutes * 60))
                    if 0 <= hour_bin < hours_per_day:
                        hourly_activity[hour_bin] += magnitude[i]
            
            # Apply smoothing if configured
            if self.timing_config.smooth_activity_data and len(hourly_activity) > 2:
                hourly_activity = self._smooth_activity_series(hourly_activity)
            
            # Check if day has sufficient active hours
            active_hours = sum(1 for activity in hourly_activity if activity > self.timing_config.min_activity_threshold)
            
            if active_hours >= self.timing_config.min_active_hours_per_day:
                daily_patterns[current_date.isoformat()] = {
                    'hourly_activity': hourly_activity,
                    'active_hours': active_hours,
                    'total_activity': sum(hourly_activity),
                    'peak_hour': hourly_activity.index(max(hourly_activity)) if hourly_activity else 0
                }
            
            current_date += timedelta(days=1)
        
        # Calculate aggregate hourly patterns
        if daily_patterns:
            aggregate_hourly = [0.0] * hours_per_day
            for day_data in daily_patterns.values():
                for i, activity in enumerate(day_data['hourly_activity']):
                    aggregate_hourly[i] += activity
            
            # Normalize by number of days
            num_days = len(daily_patterns)
            aggregate_hourly = [activity / num_days for activity in aggregate_hourly]
        else:
            aggregate_hourly = [0.0] * hours_per_day
        
        return {
            'daily_patterns': daily_patterns,
            'aggregate_hourly_pattern': aggregate_hourly,
            'hours_per_day': hours_per_day,
            'num_days_analyzed': len(daily_patterns)
        }
    
    def _smooth_activity_series(self, activity_series: List[float]) -> List[float]:
        """Apply simple moving average smoothing to activity series."""
        if len(activity_series) < 3:
            return activity_series
        
        smoothed = []
        window_size = 3
        
        for i in range(len(activity_series)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(activity_series), i + window_size // 2 + 1)
            
            window_avg = sum(activity_series[start_idx:end_idx]) / (end_idx - start_idx)
            smoothed.append(window_avg)
        
        return smoothed
    
    def _calculate_timing_variance(self, 
                                  timing_patterns: Dict[str, Any], 
                                  timestamps: List[datetime]) -> Dict[str, Any]:
        """Calculate timing variance metrics."""
        daily_patterns = timing_patterns['daily_patterns']
        
        if not daily_patterns:
            return {
                'weekly_variance': 0.0,
                'hourly_patterns': {},
                'daily_variances': {},
                'variance_by_day_type': {},
                'peak_activity_hour': None,
                'variance_stability': 0.0
            }
        
        # Calculate variance for each hour across days
        hours_per_day = timing_patterns['hours_per_day']
        hourly_variances = []
        hourly_std_devs = []
        
        for hour in range(hours_per_day):
            hour_activities = []
            for day_data in daily_patterns.values():
                if hour < len(day_data['hourly_activity']):
                    hour_activities.append(day_data['hourly_activity'][hour])
            
            if hour_activities and len(hour_activities) > 1:
                mean_activity = sum(hour_activities) / len(hour_activities)
                variance = sum((x - mean_activity)**2 for x in hour_activities) / len(hour_activities)
                std_dev = math.sqrt(variance)
                
                hourly_variances.append(variance)
                hourly_std_devs.append(std_dev)
            else:
                hourly_variances.append(0.0)
                hourly_std_devs.append(0.0)
        
        # Calculate overall weekly variance
        if self.timing_config.variance_metric == "std":
            weekly_variance = sum(hourly_std_devs) / len(hourly_std_devs) if hourly_std_devs else 0.0
        elif self.timing_config.variance_metric == "cv":
            # Coefficient of variation
            mean_variance = sum(hourly_variances) / len(hourly_variances) if hourly_variances else 0.0
            std_variance = math.sqrt(sum((v - mean_variance)**2 for v in hourly_variances) / len(hourly_variances)) if hourly_variances and len(hourly_variances) > 1 else 0.0
            weekly_variance = std_variance / mean_variance if mean_variance > 0 else 0.0
        else:  # "iqr" or default
            sorted_variances = sorted(hourly_variances)
            n = len(sorted_variances)
            if n >= 4:
                q1 = sorted_variances[n // 4]
                q3 = sorted_variances[3 * n // 4]
                weekly_variance = q3 - q1
            else:
                weekly_variance = sum(hourly_variances) / len(hourly_variances) if hourly_variances else 0.0
        
        # Calculate daily variances
        daily_variances = {}
        for date, day_data in daily_patterns.items():
            hourly_activity = day_data['hourly_activity']
            if len(hourly_activity) > 1:
                mean_activity = sum(hourly_activity) / len(hourly_activity)
                daily_variance = sum((x - mean_activity)**2 for x in hourly_activity) / len(hourly_activity)
                daily_variances[date] = daily_variance
            else:
                daily_variances[date] = 0.0
        
        # Separate weekday/weekend analysis
        variance_by_day_type = {}
        if self.timing_config.include_weekend_analysis:
            weekday_patterns = []
            weekend_patterns = []
            
            for date_str, day_data in daily_patterns.items():
                date = datetime.fromisoformat(date_str)
                if date.weekday() < 5:  # Monday-Friday
                    weekday_patterns.append(day_data['hourly_activity'])
                else:  # Saturday-Sunday
                    weekend_patterns.append(day_data['hourly_activity'])
            
            # Calculate variance for each type
            variance_by_day_type['weekday'] = self._calculate_pattern_variance(weekday_patterns)
            variance_by_day_type['weekend'] = self._calculate_pattern_variance(weekend_patterns)
            variance_by_day_type['weekday_weekend_difference'] = (
                variance_by_day_type.get('weekday', 0) - variance_by_day_type.get('weekend', 0)
            )
        
        # Find peak activity hour
        aggregate_pattern = timing_patterns['aggregate_hourly_pattern']
        peak_hour = aggregate_pattern.index(max(aggregate_pattern)) if aggregate_pattern else None
        
        # Calculate variance stability (consistency of variance across hours)
        if hourly_variances and len(hourly_variances) > 1:
            mean_variance = sum(hourly_variances) / len(hourly_variances)
            variance_stability = 1.0 - (math.sqrt(sum((v - mean_variance)**2 for v in hourly_variances) / len(hourly_variances)) / mean_variance) if mean_variance > 0 else 0
        else:
            variance_stability = 0.0
        
        # Normalize by activity level if configured
        if self.timing_config.normalize_by_activity_level:
            total_activity = sum(sum(day_data['hourly_activity']) for day_data in daily_patterns.values())
            if total_activity > 0:
                weekly_variance = weekly_variance / (total_activity / len(daily_patterns)) if daily_patterns else 0.0
        
        return {
            'weekly_variance': weekly_variance,
            'hourly_patterns': {
                'hourly_variances': hourly_variances,
                'hourly_std_devs': hourly_std_devs,
                'aggregate_pattern': aggregate_pattern
            },
            'daily_variances': daily_variances,
            'variance_by_day_type': variance_by_day_type,
            'peak_activity_hour': peak_hour,
            'variance_stability': variance_stability,
            'mean_hourly_variance': sum(hourly_variances) / len(hourly_variances) if hourly_variances else 0.0,
            'max_hourly_variance': max(hourly_variances) if hourly_variances else 0.0,
            'min_hourly_variance': min(hourly_variances) if hourly_variances else 0.0
        }
    
    def _calculate_pattern_variance(self, patterns: List[List[float]]) -> float:
        """Calculate variance for a list of activity patterns."""
        if not patterns or len(patterns) < 2:
            return 0.0
        
        # Calculate variance for each hour across patterns, then average
        num_hours = len(patterns[0])
        hourly_variances = []
        
        for hour in range(num_hours):
            hour_values = [pattern[hour] for pattern in patterns if hour < len(pattern)]
            
            if hour_values and len(hour_values) > 1:
                mean_value = sum(hour_values) / len(hour_values)
                variance = sum((x - mean_value)**2 for x in hour_values) / len(hour_values)
                hourly_variances.append(variance)
        
        return sum(hourly_variances) / len(hourly_variances) if hourly_variances else 0.0
    
    # Helper methods for app usage breadth
    def _validate_app_usage_data(self, data: Dict[str, Any]) -> None:
        """Validate app usage data format and content."""
        required_columns = ['timestamp', 'app_name', 'duration_seconds']
        
        for col in required_columns:
            if col not in data:
                raise ValueError(f"Missing required column: {col}")
        
        # Check equal lengths
        lengths = [len(data[col]) for col in required_columns]
        if len(set(lengths)) > 1:
            raise ValueError("All app usage data columns must have equal length")
        
        if len(data['timestamp']) == 0:
            raise ValueError("App usage data cannot be empty")
        
        # Validate duration values
        for duration in data['duration_seconds']:
            if not isinstance(duration, (int, float)) or duration < 0:
                raise ValueError(f"Invalid duration value: {duration}")
        
        # Validate app names
        for app_name in data['app_name']:
            if not isinstance(app_name, str) or not app_name.strip():
                raise ValueError(f"Invalid app name: {app_name}")
    
    def _filter_app_usage_by_window(self, 
                                   timestamps: List[datetime], 
                                   app_names: List[str],
                                   durations: List[float],
                                   window_start: datetime, 
                                   window_end: datetime) -> tuple:
        """Filter app usage data to specified analysis window."""
        filtered_timestamps = []
        filtered_app_names = []
        filtered_durations = []
        
        for i, ts in enumerate(timestamps):
            if window_start <= ts <= window_end:
                filtered_timestamps.append(ts)
                filtered_app_names.append(app_names[i])
                filtered_durations.append(durations[i])
        
        return filtered_timestamps, filtered_app_names, filtered_durations
    
    def _assess_app_usage_quality(self, 
                                 timestamps: List[datetime], 
                                 app_names: List[str],
                                 durations: List[float]) -> Dict[str, Any]:
        """Assess quality of app usage data."""
        if len(timestamps) < 2:
            return {
                'coverage_ratio': 0.0,
                'sessions_per_day': 0.0,
                'unique_apps': 0,
                'usage_consistency': 0.0,
                'overall_quality': 0.0
            }
        
        # Calculate sessions per day
        time_span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
        sessions_per_day = len(timestamps) / time_span_days if time_span_days > 0 else 0
        
        # Calculate unique apps
        unique_apps = len(set(app_names))
        
        # Calculate usage consistency (variance in session durations)
        if durations:
            mean_duration = sum(durations) / len(durations)
            variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
            std_duration = math.sqrt(variance)
            # Higher consistency = lower coefficient of variation
            usage_consistency = 1.0 - (std_duration / mean_duration) if mean_duration > 0 else 0
            usage_consistency = max(0, min(1, usage_consistency))
        else:
            usage_consistency = 0
        
        # Expected vs actual sessions (assuming 20 sessions/day as reference)
        expected_sessions = time_span_days * 20
        coverage_ratio = min(len(timestamps) / expected_sessions, 1.0) if expected_sessions > 0 else 0
        
        # Overall quality score
        quality_score = (
            coverage_ratio * 0.3 +
            min(1.0, sessions_per_day / 20) * 0.3 +
            min(1.0, unique_apps / 10) * 0.2 +  # 10 apps as reference
            usage_consistency * 0.2
        )
        
        return {
            'coverage_ratio': coverage_ratio,
            'sessions_per_day': sessions_per_day,
            'unique_apps': unique_apps,
            'usage_consistency': usage_consistency,
            'overall_quality': quality_score
        }
    
    def _process_app_usage_data(self, 
                               timestamps: List[datetime], 
                               app_names: List[str],
                               durations: List[float]) -> Dict[str, Any]:
        """Process and aggregate app usage data."""
        # Filter out very short sessions
        filtered_data = []
        for i, duration in enumerate(durations):
            if duration >= self.app_usage_config.min_usage_duration_seconds:
                filtered_data.append({
                    'timestamp': timestamps[i],
                    'app_name': app_names[i],
                    'duration': duration
                })
        
        # Exclude system apps if configured
        if self.app_usage_config.exclude_system_apps:
            system_apps = {'android', 'ios', 'system', 'settings', 'phone', 'messages'}
            filtered_data = [
                d for d in filtered_data 
                if d['app_name'].lower() not in system_apps
            ]
        
        # Aggregate by app
        app_usage = {}
        for entry in filtered_data:
            app = entry['app_name']
            if app not in app_usage:
                app_usage[app] = {
                    'sessions': 0,
                    'total_duration': 0.0,
                    'timestamps': []
                }
            
            app_usage[app]['sessions'] += 1
            app_usage[app]['total_duration'] += entry['duration']
            app_usage[app]['timestamps'].append(entry['timestamp'])
        
        # Filter apps by minimum session count
        filtered_apps = {
            app: data for app, data in app_usage.items()
            if data['sessions'] >= self.app_usage_config.min_app_sessions
        }
        
        return {
            'app_usage': filtered_apps,
            'total_apps': len(filtered_apps),
            'total_sessions': sum(data['sessions'] for data in filtered_apps.values()),
            'total_duration': sum(data['total_duration'] for data in filtered_apps.values())
        }
    
    def _calculate_app_usage_breadth(self, 
                                   processed_usage: Dict[str, Any], 
                                   timestamps: List[datetime]) -> Dict[str, Any]:
        """Calculate Shannon entropy of app usage patterns."""
        app_usage = processed_usage['app_usage']
        
        if not app_usage:
            return {
                'weekly_entropy': 0.0,
                'unique_apps': 0,
                'total_sessions': 0,
                'total_usage_time': 0.0,
                'app_usage_patterns': {},
                'dominant_app': None,
                'entropy_per_app': 0.0
            }
        
        # Calculate usage probabilities
        total_weight = 0.0
        app_probabilities = {}
        
        if self.app_usage_config.include_duration_weighting:
            # Weight by total duration
            total_weight = sum(data['total_duration'] for data in app_usage.values())
        else:
            # Weight by session count
            total_weight = sum(data['sessions'] for data in app_usage.values())
        
        if total_weight == 0:
            return {
                'weekly_entropy': 0.0,
                'unique_apps': len(app_usage),
                'total_sessions': processed_usage['total_sessions'],
                'total_usage_time': processed_usage['total_duration'],
                'app_usage_patterns': {},
                'dominant_app': None,
                'entropy_per_app': 0.0
            }
        
        # Calculate probabilities
        for app, data in app_usage.items():
            if self.app_usage_config.include_duration_weighting:
                probability = data['total_duration'] / total_weight
            else:
                probability = data['sessions'] / total_weight
            
            app_probabilities[app] = {
                'probability': probability,
                'sessions': data['sessions'],
                'total_duration': data['total_duration'],
                'avg_session_duration': data['total_duration'] / data['sessions'] if data['sessions'] > 0 else 0
            }
        
        # Calculate Shannon entropy
        entropy = 0.0
        for prob_info in app_probabilities.values():
            p = prob_info['probability']
            if p > 0:
                entropy -= p * math.log(p, self.app_usage_config.entropy_base)
        
        # Find dominant app
        dominant_app = max(app_probabilities.keys(), key=lambda x: app_probabilities[x]['probability']) if app_probabilities else None
        
        # Prepare app usage patterns
        app_usage_patterns = {}
        for app, prob_info in app_probabilities.items():
            app_usage_patterns[app] = {
                'usage_probability': prob_info['probability'],
                'session_count': prob_info['sessions'],
                'total_duration_seconds': prob_info['total_duration'],
                'avg_session_duration_seconds': prob_info['avg_session_duration']
            }
        
        return {
            'weekly_entropy': entropy,
            'unique_apps': len(app_usage),
            'total_sessions': processed_usage['total_sessions'],
            'total_usage_time': processed_usage['total_duration'],
            'app_usage_patterns': app_usage_patterns,
            'dominant_app': dominant_app,
            'entropy_per_app': entropy / len(app_usage) if app_usage else 0,
            'max_possible_entropy': math.log(len(app_usage), self.app_usage_config.entropy_base) if app_usage else 0.0
        }
    
    # Helper methods for activity volume
    def _validate_accelerometer_data(self, data: Dict[str, Any]) -> None:
        """Validate accelerometer data format and content."""
        required_columns = ['timestamp', 'x', 'y', 'z']
        
        for col in required_columns:
            if col not in data:
                raise ValueError(f"Missing required column: {col}")
        
        # Check equal lengths
        lengths = [len(data[col]) for col in required_columns]
        if len(set(lengths)) > 1:
            raise ValueError("All accelerometer data columns must have equal length")
        
        if len(data['timestamp']) == 0:
            raise ValueError("Accelerometer data cannot be empty")
    
    def _ensure_datetime(self, timestamps: List[Union[datetime, str]]) -> List[datetime]:
        """Convert timestamps to datetime objects."""
        datetime_timestamps = []
        
        for ts in timestamps:
            if isinstance(ts, datetime):
                datetime_timestamps.append(ts)
            elif isinstance(ts, str):
                try:
                    # Simple ISO format parsing
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    datetime_timestamps.append(dt)
                except ValueError:
                    raise ValueError(f"Invalid timestamp format: {ts}")
            else:
                raise ValueError(f"Unsupported timestamp type: {type(ts)}")
        
        return datetime_timestamps
    
    def _compute_magnitude(self, x: List[float], y: List[float], z: List[float]) -> List[float]:
        """Compute accelerometer magnitude from x, y, z components."""
        magnitude = []
        
        for i in range(len(x)):
            mag = math.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
            magnitude.append(mag)
        
        return magnitude
    
    def _filter_by_window(self, 
                         timestamps: List[datetime], 
                         magnitude: List[float],
                         window_start: datetime, 
                         window_end: datetime) -> tuple:
        """Filter data to specified analysis window."""
        filtered_timestamps = []
        filtered_magnitude = []
        
        for i, ts in enumerate(timestamps):
            if window_start <= ts <= window_end:
                filtered_timestamps.append(ts)
                filtered_magnitude.append(magnitude[i])
        
        return filtered_timestamps, filtered_magnitude
    
    def _assess_data_quality(self, 
                            timestamps: List[datetime], 
                            magnitude: List[float]) -> Dict[str, Any]:
        """Assess quality of accelerometer data."""
        if len(timestamps) < 2:
            return {
                'coverage_ratio': 0.0,
                'sampling_rate_hz': 0.0,
                'gap_statistics': {'max_gap_minutes': float('inf'), 'total_gap_minutes': float('inf')},
                'outlier_statistics': {'outlier_count': 0, 'outlier_percentage': 0.0},
                'overall_quality': 0.0
            }
        
        # Calculate sampling rate
        time_span = (max(timestamps) - min(timestamps)).total_seconds()
        sampling_rate_hz = len(timestamps) / time_span if time_span > 0 else 0
        
        # Calculate gap statistics
        gaps = []
        for i in range(1, len(timestamps)):
            gap_minutes = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
            gaps.append(gap_minutes)
        
        max_gap_minutes = max(gaps) if gaps else 0
        total_gap_minutes = sum(gaps)
        
        # Expected vs actual sampling
        expected_samples = time_span * sampling_rate_hz
        coverage_ratio = min(len(timestamps) / expected_samples, 1.0) if expected_samples > 0 else 0
        
        # Outlier detection (simple z-score method)
        outlier_count = 0
        if len(magnitude) > 1:
            mean_mag = sum(magnitude) / len(magnitude)
            variance = sum((x - mean_mag) ** 2 for x in magnitude) / len(magnitude)
            std_mag = math.sqrt(variance)
            
            if std_mag > 0:
                for mag in magnitude:
                    z_score = abs(mag - mean_mag) / std_mag
                    if z_score > self.activity_config.outlier_threshold_std:
                        outlier_count += 1
        
        outlier_percentage = outlier_count / len(magnitude) if magnitude else 0
        
        # Overall quality score (0-1)
        quality_score = (
            coverage_ratio * 0.4 +
            min(1.0, sampling_rate_hz / self.activity_config.min_sampling_rate_hz) * 0.3 +
            min(1.0, self.activity_config.max_gap_minutes / max(max_gap_minutes, 1)) * 0.2 +
            (1 - outlier_percentage) * 0.1
        )
        
        return {
            'coverage_ratio': coverage_ratio,
            'sampling_rate_hz': sampling_rate_hz,
            'gap_statistics': {
                'max_gap_minutes': max_gap_minutes,
                'total_gap_minutes': total_gap_minutes
            },
            'outlier_statistics': {
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_percentage
            },
            'overall_quality': quality_score
        }
    
    def _preprocess_data(self, 
                         timestamps: List[datetime], 
                         magnitude: List[float],
                         quality_metrics: Dict[str, Any]) -> List[float]:
        """Preprocess data with interpolation and outlier removal."""
        processed_magnitude = magnitude.copy()
        
        # Outlier removal
        if self.activity_config.remove_outliers and quality_metrics['outlier_statistics']['outlier_count'] > 0:
            processed_magnitude = self._remove_outliers(processed_magnitude)
        
        # Interpolation for gaps (simplified linear interpolation)
        if self.activity_config.interpolate_gaps and quality_metrics['gap_statistics']['max_gap_minutes'] > 0:
            processed_magnitude = self._interpolate_gaps(
                timestamps, processed_magnitude, quality_metrics
            )
        
        return processed_magnitude
    
    def _remove_outliers(self, magnitude: List[float]) -> List[float]:
        """Remove outliers using z-score method."""
        if len(magnitude) < 3:
            return magnitude
        
        mean_mag = sum(magnitude) / len(magnitude)
        variance = sum((x - mean_mag) ** 2 for x in magnitude) / len(magnitude)
        std_mag = math.sqrt(variance)
        
        if std_mag == 0:
            return magnitude
        
        processed = []
        for mag in magnitude:
            z_score = abs(mag - mean_mag) / std_mag
            if z_score <= self.activity_config.outlier_threshold_std:
                processed.append(mag)
            else:
                # Replace with mean instead of removing
                processed.append(mean_mag)
        
        return processed
    
    def _interpolate_gaps(self, 
                         timestamps: List[datetime], 
                         magnitude: List[float],
                         quality_metrics: Dict[str, Any]) -> List[float]:
        """Simple linear interpolation for small gaps."""
        # Mock implementation - in real implementation would use pandas interpolation
        warnings.warn("Using mock interpolation implementation")
        return magnitude
    
    def _calculate_daily_volumes(self, 
                                timestamps: List[datetime], 
                                magnitude: List[float],
                                window_start: datetime,
                                window_end: datetime) -> List[Dict[str, Any]]:
        """Calculate activity volume for each 24-hour window."""
        daily_volumes = []
        
        # Group data by day
        current_date = window_start.date()
        end_date = window_end.date()
        
        while current_date <= end_date:
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = day_start + timedelta(hours=24)
            
            # Filter data for this day
            day_magnitude = []
            for i, ts in enumerate(timestamps):
                if day_start <= ts < day_end:
                    day_magnitude.append(magnitude[i])
            
            # Calculate activity volume (sum of magnitude)
            if day_magnitude:
                daily_volume = sum(day_magnitude)
                duration_hours = min(24, (day_end - day_start).total_seconds() / 3600)
                volume_per_hour = daily_volume / duration_hours if duration_hours > 0 else 0
            else:
                daily_volume = 0
                volume_per_hour = 0
            
            daily_volumes.append({
                'timestamp': day_start.isoformat(),
                'date': current_date.isoformat(),
                'volume': daily_volume,
                'volume_per_hour': volume_per_hour,
                'sample_count': len(day_magnitude),
                'window_start': day_start.isoformat(),
                'window_end': day_end.isoformat()
            })
            
            current_date += timedelta(days=1)
        
        return daily_volumes
    
    # Helper methods for location diversity
    
    def _validate_gps_data(self, data: Dict[str, Any]) -> None:
        """Validate GPS data format and content."""
        required_columns = ['timestamp', 'latitude', 'longitude']
        
        for col in required_columns:
            if col not in data:
                raise ValueError(f"Missing required column: {col}")
        
        # Check equal lengths
        lengths = [len(data[col]) for col in required_columns]
        if len(set(lengths)) > 1:
            raise ValueError("All GPS data columns must have equal length")
        
        if len(data['timestamp']) == 0:
            raise ValueError("GPS data cannot be empty")
        
        # Validate coordinate ranges
        for lat in data['latitude']:
            if not (-90 <= lat <= 90):
                raise ValueError(f"Invalid latitude value: {lat}")
        
        for lon in data['longitude']:
            if not (-180 <= lon <= 180):
                raise ValueError(f"Invalid longitude value: {lon}")
    
    def _filter_gps_by_window(self, 
                             timestamps: List[datetime], 
                             latitudes: List[float],
                             longitudes: List[float],
                             window_start: datetime, 
                             window_end: datetime) -> tuple:
        """Filter GPS data to specified analysis window."""
        filtered_timestamps = []
        filtered_latitudes = []
        filtered_longitudes = []
        
        for i, ts in enumerate(timestamps):
            if window_start <= ts <= window_end:
                filtered_timestamps.append(ts)
                filtered_latitudes.append(latitudes[i])
                filtered_longitudes.append(longitudes[i])
        
        return filtered_timestamps, filtered_latitudes, filtered_longitudes
    
    def _assess_gps_quality(self, 
                           timestamps: List[datetime], 
                           latitudes: List[float],
                           longitudes: List[float]) -> Dict[str, Any]:
        """Assess quality of GPS data."""
        if len(timestamps) < 2:
            return {
                'coverage_ratio': 0.0,
                'sampling_rate_per_day': 0.0,
                'geographic_area': 0.0,
                'coordinate_consistency': 0.0,
                'overall_quality': 0.0
            }
        
        # Calculate sampling rate
        time_span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
        sampling_rate_per_day = len(timestamps) / time_span_days if time_span_days > 0 else 0
        
        # Calculate geographic area (bounding box)
        min_lat, max_lat = min(latitudes), max(latitudes)
        min_lon, max_lon = min(longitudes), max(longitudes)
        
        # Approximate area calculation (simplified)
        lat_diff = max_lat - min_lat
        lon_diff = max_lon - min_lon
        geographic_area = lat_diff * lon_diff * 111000 * 111000  # Approximate square meters
        
        # Coordinate consistency (check for unrealistic jumps)
        coordinate_jumps = 0
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            if time_diff > 0:
                lat_diff = abs(latitudes[i] - latitudes[i-1])
                lon_diff = abs(longitudes[i] - longitudes[i-1])
                
                # Check for unrealistic jumps (>100km in 1 minute)
                if time_diff < 60 and (lat_diff > 0.9 or lon_diff > 0.9):
                    coordinate_jumps += 1
        
        coordinate_consistency = 1.0 - (coordinate_jumps / len(timestamps)) if timestamps else 0
        
        # Expected vs actual sampling
        expected_samples = time_span_days * 24 * 60  # Assuming 1 sample per minute
        coverage_ratio = min(len(timestamps) / expected_samples, 1.0) if expected_samples > 0 else 0
        
        # Overall quality score
        quality_score = (
            coverage_ratio * 0.3 +
            min(1.0, sampling_rate_per_day / 100) * 0.3 +
            min(1.0, geographic_area / 1000000) * 0.2 +  # 1 km² reference
            coordinate_consistency * 0.2
        )
        
        return {
            'coverage_ratio': coverage_ratio,
            'sampling_rate_per_day': sampling_rate_per_day,
            'geographic_area': geographic_area,
            'coordinate_consistency': coordinate_consistency,
            'overall_quality': quality_score
        }
    
    def _perform_location_clustering(self, 
                                   latitudes: List[float], 
                                   longitudes: List[float],
                                   quality_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform location clustering using radius-based approach."""
        clusters = []
        used_indices = set()
        
        # Convert radius to degrees (approximate)
        radius_degrees = self.location_config.clustering_radius_meters / 111000
        
        for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
            if i in used_indices:
                continue
            
            # Find all points within radius
            cluster_points = []
            for j, (lat2, lon2) in enumerate(zip(latitudes, longitudes)):
                if j not in used_indices:
                    distance = self._haversine_distance(lat, lon, lat2, lon2)
                    if distance <= self.location_config.clustering_radius_meters:
                        cluster_points.append((j, lat2, lon2))
                        used_indices.add(j)
            
            # Create cluster if minimum size reached
            if len(cluster_points) >= self.location_config.min_cluster_size:
                cluster_lat = sum(p[1] for p in cluster_points) / len(cluster_points)
                cluster_lon = sum(p[2] for p in cluster_points) / len(cluster_points)
                
                cluster = {
                    'cluster_id': len(clusters),
                    'center_latitude': cluster_lat,
                    'center_longitude': cluster_lon,
                    'point_count': len(cluster_points),
                    'point_indices': [p[0] for p in cluster_points],
                    'radius_meters': self._calculate_cluster_radius(cluster_points),
                    'is_home': False  # Will be determined later
                }
                clusters.append(cluster)
        
        # Identify home cluster (most visited, likely at night)
        if clusters and self.location_config.remove_home_location:
            home_cluster = self._identify_home_cluster(clusters, latitudes, longitudes)
            if home_cluster:
                home_cluster['is_home'] = True
        
        return clusters
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two GPS coordinates."""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _calculate_cluster_radius(self, cluster_points: List[Tuple[int, float, float]]) -> float:
        """Calculate the radius of a cluster."""
        if len(cluster_points) <= 1:
            return 0.0
        
        # Use center as reference
        center_lat = sum(p[1] for p in cluster_points) / len(cluster_points)
        center_lon = sum(p[2] for p in cluster_points) / len(cluster_points)
        
        max_distance = 0.0
        for _, lat, lon in cluster_points:
            distance = self._haversine_distance(center_lat, center_lon, lat, lon)
            max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _identify_home_cluster(self, 
                             clusters: List[Dict[str, Any]], 
                             latitudes: List[float],
                             longitudes: List[float]) -> Optional[Dict[str, Any]]:
        """Identify the most likely home cluster."""
        if not clusters:
            return None
        
        # Score clusters based on visitation patterns
        cluster_scores = []
        
        for cluster in clusters:
            score = 0
            
            # Check points during typical sleep hours (10 PM - 6 AM)
            for idx in cluster['point_indices']:
                # Note: This would need timestamp data for proper implementation
                # For now, use point count as a simple proxy
                score += 1
            
            # Prefer larger clusters
            score += cluster['point_count'] * 0.1
            
            cluster_scores.append((cluster, score))
        
        # Return cluster with highest score
        if cluster_scores:
            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            return cluster_scores[0][0]
        
        return None
    
    def _calculate_location_diversity(self, 
                                    clusters: List[Dict[str, Any]], 
                                    timestamps: List[datetime]) -> Dict[str, Any]:
        """Calculate Shannon entropy of location visits."""
        if not clusters:
            return {
                'weekly_entropy': 0.0,
                'cluster_count': 0,
                'unique_locations': 0,
                'location_probabilities': {},
                'home_cluster_removed': False
            }
        
        # Remove home cluster if configured
        active_clusters = clusters
        home_removed = False
        
        if self.location_config.remove_home_location:
            home_clusters = [c for c in clusters if c.get('is_home', False)]
            if home_clusters:
                active_clusters = [c for c in clusters if not c.get('is_home', False)]
                home_removed = True
        
        if not active_clusters:
            return {
                'weekly_entropy': 0.0,
                'cluster_count': 0,
                'unique_locations': 0,
                'location_probabilities': {},
                'home_cluster_removed': home_removed
            }
        
        # Calculate visitation probabilities
        total_visits = sum(c['point_count'] for c in active_clusters)
        location_probabilities = {}
        
        for cluster in active_clusters:
            probability = cluster['point_count'] / total_visits
            location_probabilities[cluster['cluster_id']] = {
                'probability': probability,
                'visit_count': cluster['point_count'],
                'center_lat': cluster['center_latitude'],
                'center_lon': cluster['center_longitude']
            }
        
        # Calculate Shannon entropy
        entropy = 0.0
        for prob_info in location_probabilities.values():
            p = prob_info['probability']
            if p > 0:
                entropy -= p * math.log(p, self.location_config.entropy_base)
        
        return {
            'weekly_entropy': entropy,
            'cluster_count': len(active_clusters),
            'unique_locations': len(active_clusters),
            'location_probabilities': location_probabilities,
            'home_cluster_removed': home_removed,
            'max_possible_entropy': math.log(len(active_clusters), self.location_config.entropy_base) if active_clusters else 0.0
        }
