"""
Avoidance (AV) construct features.

This module implements features related to behavioral avoidance patterns,
including home confinement, communication gaps, and movement limitations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import math


@dataclass
class HomeConfinementConfig:
    """Configuration for home confinement feature extraction."""
    
    # Home cluster parameters
    home_radius_meters: float = 50.0  # Radius for home cluster detection
    min_night_points: int = 10  # Minimum nighttime points for home detection
    night_start_hour: int = 22  # 10 PM
    night_end_hour: int = 6   # 6 AM
    
    # Analysis parameters
    analysis_window_days: int = 7  # Weekly analysis
    min_gps_points: int = 100  # Minimum GPS points required
    
    # Quality thresholds
    min_data_coverage: float = 0.6  # 60% minimum coverage
    max_gap_hours: float = 4.0  # Maximum gap in data


@dataclass
class CommunicationGapsConfig:
    """Configuration for communication gaps feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 7  # Weekly analysis
    min_communications: int = 3  # Minimum communications per day
    
    # Gap detection parameters
    max_gap_hours: float = 24.0  # Maximum gap to consider
    min_gap_duration_minutes: float = 30.0  # Minimum gap to count
    
    # Data quality requirements
    min_days_with_data: int = 5  # Minimum days with communication data
    min_data_coverage: float = 0.5  # 50% minimum coverage


@dataclass
class MovementRadiusConfig:
    """Configuration for movement radius feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 7  # Weekly analysis
    min_gps_points: int = 50  # Minimum GPS points
    
    # Radius calculation parameters
    use_haversine: bool = True  # Use haversine distance vs euclidean
    outlier_threshold_std: float = 3.0  # Outlier detection threshold
    
    # Quality requirements
    min_data_coverage: float = 0.6  # 60% minimum coverage
    coordinate_precision: float = 1e-6  # Minimum coordinate precision


class AvoidanceFeatures:
    """
    Implementation of Avoidance (AV) construct features.
    
    This class provides methods for extracting features related to behavioral
    avoidance, which reflects the tendency to avoid certain situations,
    locations, or social interactions.
    
    Attributes:
        home_config: Configuration for home confinement features
        comm_config: Configuration for communication gaps features
        radius_config: Configuration for movement radius features
        provenance_tracker: Provenance tracking instance
    """
    
    def __init__(self, 
                 home_config: Optional[HomeConfinementConfig] = None,
                 comm_config: Optional[CommunicationGapsConfig] = None,
                 radius_config: Optional[MovementRadiusConfig] = None):
        """
        Initialize avoidance features extractor.
        
        Args:
            home_config: Configuration for home confinement features
            comm_config: Configuration for communication gaps features
            radius_config: Configuration for movement radius features
        """
        self.home_config = home_config or HomeConfinementConfig()
        self.comm_config = comm_config or CommunicationGapsConfig()
        self.radius_config = radius_config or MovementRadiusConfig()
        
        # Import provenance tracker locally to avoid circular imports
        try:
            from ..utils.provenance import get_provenance_tracker
            self.provenance_tracker = get_provenance_tracker()
        except ImportError:
            self.provenance_tracker = None
    
    def home_confinement(self,
                        gps_data: Dict[str, Any],
                        window_start: Optional[datetime] = None,
                        window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate home confinement from GPS data.
        
        Feature Name: Home Confinement
        Construct: Avoidance (AV)
        Mathematical Definition: Percentage of GPS readings within defined home cluster radius
        Formal Equation: HC = (points_within_home / total_points) * 100
        Assumptions: Home location can be identified from nighttime GPS clustering
        Limitations: Sensitive to GPS accuracy, home detection algorithm, and travel patterns
        Edge Cases: No clear home cluster, insufficient nighttime data, extensive travel
        Output Schema: Weekly home confinement percentage with home location and quality metrics
        
        Args:
            gps_data: Dictionary with 'timestamp', 'latitude', 'longitude' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing home confinement values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_home_confinement",
                input_parameters={
                    "home_radius_meters": self.home_config.home_radius_meters,
                    "night_start_hour": self.home_config.night_start_hour,
                    "night_end_hour": self.home_config.night_end_hour,
                    "analysis_window_days": self.home_config.analysis_window_days
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
            if len(filtered_timestamps) < self.home_config.min_gps_points:
                raise ValueError(
                    f"Insufficient GPS points: {len(filtered_timestamps)} "
                    f"< {self.home_config.min_gps_points}"
                )
            
            # Quality assessment
            quality_metrics = self._assess_gps_quality(
                filtered_timestamps, filtered_latitudes, filtered_longitudes
            )
            
            # Identify home location
            home_location = self._identify_home_location(
                filtered_timestamps, filtered_latitudes, filtered_longitudes
            )
            
            # Calculate home confinement
            confinement_metrics = self._calculate_home_confinement(
                filtered_latitudes, filtered_longitudes, home_location
            )
            
            # Prepare results
            result = {
                'home_confinement': confinement_metrics,
                'weekly_confinement_percentage': confinement_metrics['confinement_percentage'],
                'home_location': home_location,
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'home_radius_meters': self.home_config.home_radius_meters,
                    'night_start_hour': self.home_config.night_start_hour,
                    'night_end_hour': self.home_config.night_end_hour,
                    'analysis_window_days': self.home_config.analysis_window_days,
                    'min_night_points': self.home_config.min_night_points
                },
                'data_summary': {
                    'total_gps_points': len(filtered_timestamps),
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days,
                    'nighttime_points_used': home_location.get('nighttime_points_count', 0)
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'confinement_percentage': confinement_metrics['confinement_percentage'],
                        'home_detected': home_location.get('detected', False),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="home_confinement",
                    construct="avoidance",
                    input_data_summary={
                        'gps_points': len(filtered_timestamps),
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400,
                        'nighttime_points': home_location.get('nighttime_points_count', 0)
                    },
                    computation_parameters={
                        'home_radius_meters': self.home_config.home_radius_meters,
                        'night_start_hour': self.home_config.night_start_hour,
                        'night_end_hour': self.home_config.night_end_hour
                    },
                    result_summary={
                        'confinement_percentage': confinement_metrics['confinement_percentage'],
                        'points_within_home': confinement_metrics['points_within_home'],
                        'home_latitude': home_location.get('latitude', 'N/A'),
                        'home_longitude': home_location.get('longitude', 'N/A')
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
    
    def communication_gaps(self,
                          communication_data: Dict[str, Any],
                          window_start: Optional[datetime] = None,
                          window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate communication gaps from communication logs.
        
        Feature Name: Communication Gaps
        Construct: Avoidance (AV)
        Mathematical Definition: Maximum duration without outgoing communication per day
        Formal Equation: CG = max(t_i+1 - t_i) where t_i are outgoing communication timestamps
        Assumptions: Communication data represents meaningful social interaction attempts
        Limitations: Sensitive to communication platform coverage and usage patterns
        Edge Cases: No outgoing communications, single communication per day, platform switching
        Output Schema: Daily longest silence periods with weekly statistics and quality metrics
        
        Args:
            communication_data: Dictionary with 'timestamp', 'direction', 'contact' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing communication gaps values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_communication_gaps",
                input_parameters={
                    "analysis_window_days": self.comm_config.analysis_window_days,
                    "min_gap_duration_minutes": self.comm_config.min_gap_duration_minutes,
                    "max_gap_hours": self.comm_config.max_gap_hours,
                    "min_communications": self.comm_config.min_communications
                }
            )
        
        try:
            # Validate input data
            self._validate_communication_data(communication_data)
            
            # Extract and preprocess communication data
            timestamps = self._ensure_datetime(communication_data['timestamp'])
            directions = communication_data['direction']
            contacts = communication_data['contact']
            
            # Set analysis window
            if window_start is None:
                window_start = min(timestamps)
            if window_end is None:
                window_end = max(timestamps)
            
            # Filter data to analysis window
            filtered_data = self._filter_communication_by_window(
                timestamps, directions, contacts, window_start, window_end
            )
            filtered_timestamps, filtered_directions, filtered_contacts = filtered_data
            
            # Check minimum data requirements
            if len(filtered_timestamps) < self.comm_config.min_communications:
                raise ValueError(
                    f"Insufficient communication data: {len(filtered_timestamps)} "
                    f"< {self.comm_config.min_communications}"
                )
            
            # Quality assessment
            quality_metrics = self._assess_communication_quality(
                filtered_timestamps, filtered_directions, filtered_contacts
            )
            
            # Calculate daily communication gaps
            gap_metrics = self._calculate_communication_gaps(
                filtered_timestamps, filtered_directions, window_start, window_end
            )
            
            # Prepare results
            result = {
                'communication_gaps': gap_metrics,
                'weekly_max_gap_hours': gap_metrics['weekly_max_gap_hours'],
                'daily_gaps': gap_metrics['daily_gaps'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.comm_config.analysis_window_days,
                    'min_gap_duration_minutes': self.comm_config.min_gap_duration_minutes,
                    'max_gap_hours': self.comm_config.max_gap_hours,
                    'min_communications': self.comm_config.min_communications
                },
                'data_summary': {
                    'total_communications': len(filtered_timestamps),
                    'outgoing_communications': gap_metrics['total_outgoing'],
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days,
                    'days_with_data': len(gap_metrics['daily_gaps'])
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'weekly_max_gap_hours': gap_metrics['weekly_max_gap_hours'],
                        'days_analyzed': len(gap_metrics['daily_gaps']),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="communication_gaps",
                    construct="avoidance",
                    input_data_summary={
                        'communications': len(filtered_timestamps),
                        'outgoing': gap_metrics['total_outgoing'],
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400
                    },
                    computation_parameters={
                        'min_gap_duration_minutes': self.comm_config.min_gap_duration_minutes,
                        'max_gap_hours': self.comm_config.max_gap_hours
                    },
                    result_summary={
                        'weekly_max_gap_hours': gap_metrics['weekly_max_gap_hours'],
                        'mean_daily_gap': gap_metrics['mean_daily_gap_hours'],
                        'days_with_no_outgoing': gap_metrics['days_with_no_outgoing']
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
    
    def movement_radius(self,
                       gps_data: Dict[str, Any],
                       window_start: Optional[datetime] = None,
                       window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate movement radius from GPS data.
        
        Feature Name: Movement Radius
        Construct: Avoidance (AV)
        Mathematical Definition: Radius of gyration of GPS coordinates
        Formal Equation: r_g = sqrt((1/N) * Σ(r_i - r_cm)²) where r_cm is center of mass
        Assumptions: GPS data represents meaningful movement patterns and is accurately recorded
        Limitations: Sensitive to GPS accuracy, outlier locations, and sampling frequency
        Edge Cases: Single location, insufficient GPS points, extreme outliers
        Output Schema: Weekly movement radius with center of mass and quality metrics
        
        Args:
            gps_data: Dictionary with 'timestamp', 'latitude', 'longitude' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing movement radius values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_movement_radius",
                input_parameters={
                    "analysis_window_days": self.radius_config.analysis_window_days,
                    "use_haversine": self.radius_config.use_haversine,
                    "outlier_threshold_std": self.radius_config.outlier_threshold_std,
                    "min_gps_points": self.radius_config.min_gps_points
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
            if len(filtered_timestamps) < self.radius_config.min_gps_points:
                raise ValueError(
                    f"Insufficient GPS points: {len(filtered_timestamps)} "
                    f"< {self.radius_config.min_gps_points}"
                )
            
            # Quality assessment
            quality_metrics = self._assess_gps_quality(
                filtered_timestamps, filtered_latitudes, filtered_longitudes
            )
            
            # Remove outliers if configured
            if self.radius_config.outlier_threshold_std > 0:
                filtered_latitudes, filtered_longitudes = self._remove_gps_outliers(
                    filtered_latitudes, filtered_longitudes
                )
            
            # Calculate center of mass
            center_of_mass = self._calculate_center_of_mass(
                filtered_latitudes, filtered_longitudes
            )
            
            # Calculate movement radius
            radius_metrics = self._calculate_movement_radius(
                filtered_latitudes, filtered_longitudes, center_of_mass
            )
            
            # Prepare results
            result = {
                'movement_radius': radius_metrics,
                'weekly_radius_meters': radius_metrics['radius_meters'],
                'center_of_mass': center_of_mass,
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.radius_config.analysis_window_days,
                    'use_haversine': self.radius_config.use_haversine,
                    'outlier_threshold_std': self.radius_config.outlier_threshold_std,
                    'min_gps_points': self.radius_config.min_gps_points
                },
                'data_summary': {
                    'total_gps_points': len(filtered_timestamps),
                    'outliers_removed': len(gps_data['latitude']) - len(filtered_latitudes),
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
                        'weekly_radius_meters': radius_metrics['radius_meters'],
                        'center_of_mass_detected': center_of_mass.get('detected', False),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="movement_radius",
                    construct="avoidance",
                    input_data_summary={
                        'gps_points': len(filtered_timestamps),
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400,
                        'outliers_removed': len(gps_data['latitude']) - len(filtered_latitudes)
                    },
                    computation_parameters={
                        'use_haversine': self.radius_config.use_haversine,
                        'outlier_threshold_std': self.radius_config.outlier_threshold_std
                    },
                    result_summary={
                        'weekly_radius_meters': radius_metrics['radius_meters'],
                        'center_latitude': center_of_mass.get('latitude', 'N/A'),
                        'center_longitude': center_of_mass.get('longitude', 'N/A'),
                        'max_distance_meters': radius_metrics['max_distance_meters']
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
    
    # Helper methods for home confinement
    
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
        for lat, lon in zip(data['latitude'], data['longitude']):
            if not (-90 <= lat <= 90):
                raise ValueError(f"Invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Invalid longitude: {lon}")
    
    def _validate_communication_data(self, data: Dict[str, Any]) -> None:
        """Validate communication data format and content."""
        required_columns = ['timestamp', 'direction', 'contact']
        
        for col in required_columns:
            if col not in data:
                raise ValueError(f"Missing required column: {col}")
        
        # Check equal lengths
        lengths = [len(data[col]) for col in required_columns]
        if len(set(lengths)) > 1:
            raise ValueError("All communication data columns must have equal length")
        
        if len(data['timestamp']) == 0:
            raise ValueError("Communication data cannot be empty")
        
        # Validate direction values
        valid_directions = {'incoming', 'outgoing', 'in', 'out'}
        for direction in data['direction']:
            if direction.lower() not in valid_directions:
                raise ValueError(f"Invalid direction: {direction}")
    
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
    
    def _filter_gps_by_window(self, 
                             timestamps: List[datetime], 
                             latitudes: List[float],
                             longitudes: List[float],
                             window_start: datetime, 
                             window_end: datetime) -> Tuple[List[datetime], List[float], List[float]]:
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
    
    def _filter_communication_by_window(self, 
                                       timestamps: List[datetime], 
                                       directions: List[str],
                                       contacts: List[str],
                                       window_start: datetime, 
                                       window_end: datetime) -> Tuple[List[datetime], List[str], List[str]]:
        """Filter communication data to specified analysis window."""
        filtered_timestamps = []
        filtered_directions = []
        filtered_contacts = []
        
        for i, ts in enumerate(timestamps):
            if window_start <= ts <= window_end:
                filtered_timestamps.append(ts)
                filtered_directions.append(directions[i])
                filtered_contacts.append(contacts[i])
        
        return filtered_timestamps, filtered_directions, filtered_contacts
    
    def _assess_gps_quality(self, 
                           timestamps: List[datetime], 
                           latitudes: List[float],
                           longitudes: List[float]) -> Dict[str, Any]:
        """Assess quality of GPS data."""
        if len(timestamps) < 2:
            return {
                'coverage_ratio': 0.0,
                'sampling_rate_per_day': 0.0,
                'coordinate_consistency': 0.0,
                'overall_quality': 0.0
            }
        
        # Calculate sampling rate
        time_span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
        sampling_rate_per_day = len(timestamps) / time_span_days if time_span_days > 0 else 0
        
        # Calculate coordinate consistency (check for duplicate coordinates)
        coordinate_pairs = list(zip(latitudes, longitudes))
        unique_coordinates = len(set(coordinate_pairs))
        coordinate_consistency = unique_coordinates / len(coordinate_pairs) if coordinate_pairs else 0
        
        # Calculate coverage ratio (expected vs actual points)
        expected_points = time_span_days * 24 * 60  # Assuming 1 point per minute as reference
        coverage_ratio = min(len(timestamps) / expected_points, 1.0) if expected_points > 0 else 0
        
        # Overall quality score
        quality_score = (
            coverage_ratio * 0.4 +
            min(1.0, sampling_rate_per_day / 100) * 0.3 +  # 100 points/day as reference
            coordinate_consistency * 0.3
        )
        
        return {
            'coverage_ratio': coverage_ratio,
            'sampling_rate_per_day': sampling_rate_per_day,
            'coordinate_consistency': coordinate_consistency,
            'overall_quality': quality_score
        }
    
    def _assess_communication_quality(self, 
                                     timestamps: List[datetime], 
                                     directions: List[str],
                                     contacts: List[str]) -> Dict[str, Any]:
        """Assess quality of communication data."""
        if len(timestamps) < 2:
            return {
                'coverage_ratio': 0.0,
                'communications_per_day': 0.0,
                'direction_balance': 0.0,
                'overall_quality': 0.0
            }
        
        # Calculate communications per day
        time_span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
        communications_per_day = len(timestamps) / time_span_days if time_span_days > 0 else 0
        
        # Calculate direction balance
        outgoing_count = sum(1 for d in directions if d.lower() in ['outgoing', 'out'])
        incoming_count = sum(1 for d in directions if d.lower() in ['incoming', 'in'])
        total_communications = len(directions)
        
        direction_balance = 1.0 - abs(outgoing_count - incoming_count) / total_communications if total_communications > 0 else 0
        
        # Calculate coverage ratio
        expected_communications = time_span_days * 10  # 10 communications/day as reference
        coverage_ratio = min(len(timestamps) / expected_communications, 1.0) if expected_communications > 0 else 0
        
        # Overall quality score
        quality_score = (
            coverage_ratio * 0.4 +
            min(1.0, communications_per_day / 20) * 0.3 +  # 20 communications/day as reference
            direction_balance * 0.3
        )
        
        return {
            'coverage_ratio': coverage_ratio,
            'communications_per_day': communications_per_day,
            'direction_balance': direction_balance,
            'overall_quality': quality_score
        }
    
    def _identify_home_location(self, 
                               timestamps: List[datetime], 
                               latitudes: List[float],
                               longitudes: List[float]) -> Dict[str, Any]:
        """Identify home location from nighttime GPS clustering."""
        # Filter nighttime points
        nighttime_points = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            if hour >= self.home_config.night_start_hour or hour < self.home_config.night_end_hour:
                nighttime_points.append((latitudes[i], longitudes[i]))
        
        if len(nighttime_points) < self.home_config.min_night_points:
            return {
                'detected': False,
                'reason': 'Insufficient nighttime points',
                'nighttime_points_count': len(nighttime_points)
            }
        
        # Simple clustering: find the point with most neighbors within radius
        best_center = None
        max_neighbors = 0
        
        for lat, lon in nighttime_points:
            neighbor_count = 0
            for other_lat, other_lon in nighttime_points:
                distance = self._haversine_distance(lat, lon, other_lat, other_lon)
                if distance <= self.home_config.home_radius_meters:
                    neighbor_count += 1
            
            if neighbor_count > max_neighbors:
                max_neighbors = neighbor_count
                best_center = (lat, lon)
        
        if best_center and max_neighbors >= self.home_config.min_night_points:
            return {
                'detected': True,
                'latitude': best_center[0],
                'longitude': best_center[1],
                'confidence': max_neighbors / len(nighttime_points),
                'nighttime_points_count': len(nighttime_points),
                'neighbors_in_radius': max_neighbors
            }
        else:
            return {
                'detected': False,
                'reason': 'No suitable home cluster found',
                'nighttime_points_count': len(nighttime_points)
            }
    
    def _calculate_home_confinement(self, 
                                   latitudes: List[float],
                                   longitudes: List[float],
                                   home_location: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate home confinement percentage."""
        if not home_location.get('detected', False):
            return {
                'confinement_percentage': 0.0,
                'points_within_home': 0,
                'total_points': len(latitudes),
                'home_detected': False
            }
        
        home_lat = home_location['latitude']
        home_lon = home_location['longitude']
        home_radius = self.home_config.home_radius_meters
        
        points_within_home = 0
        for lat, lon in zip(latitudes, longitudes):
            distance = self._haversine_distance(lat, lon, home_lat, home_lon)
            if distance <= home_radius:
                points_within_home += 1
        
        confinement_percentage = (points_within_home / len(latitudes)) * 100 if latitudes else 0.0
        
        return {
            'confinement_percentage': confinement_percentage,
            'points_within_home': points_within_home,
            'total_points': len(latitudes),
            'home_detected': True,
            'home_radius_meters': home_radius
        }
    
    def _calculate_communication_gaps(self, 
                                     timestamps: List[datetime],
                                     directions: List[str],
                                     window_start: datetime,
                                     window_end: datetime) -> Dict[str, Any]:
        """Calculate communication gaps between outgoing communications."""
        # Filter outgoing communications
        outgoing_timestamps = []
        for ts, direction in zip(timestamps, directions):
            if direction.lower() in ['outgoing', 'out']:
                outgoing_timestamps.append(ts)
        
        total_outgoing = len(outgoing_timestamps)
        
        # Group by day
        daily_gaps = {}
        current_date = window_start.date()
        end_date = window_end.date()
        
        while current_date <= end_date:
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = day_start + timedelta(days=1)
            
            # Get outgoing communications for this day
            day_outgoing = [ts for ts in outgoing_timestamps if day_start <= ts < day_end]
            
            if len(day_outgoing) >= 2:
                # Calculate gaps between consecutive outgoing communications
                gaps = []
                for i in range(len(day_outgoing) - 1):
                    gap_duration = (day_outgoing[i + 1] - day_outgoing[i]).total_seconds() / 3600  # hours
                    if gap_duration >= self.comm_config.min_gap_duration_minutes / 60:
                        gaps.append(gap_duration)
                
                if gaps:
                    max_gap = max(gaps)
                    mean_gap = sum(gaps) / len(gaps)
                else:
                    max_gap = 0.0
                    mean_gap = 0.0
                
                daily_gaps[current_date.isoformat()] = {
                    'max_gap_hours': max_gap,
                    'mean_gap_hours': mean_gap,
                    'outgoing_count': len(day_outgoing),
                    'gap_count': len(gaps)
                }
            elif len(day_outgoing) == 1:
                # Only one outgoing communication - check gap to day boundaries
                first_comm = day_outgoing[0]
                gap_to_start = (first_comm - day_start).total_seconds() / 3600
                gap_to_end = (day_end - first_comm).total_seconds() / 3600
                
                max_gap = max(gap_to_start, gap_to_end)
                
                daily_gaps[current_date.isoformat()] = {
                    'max_gap_hours': max_gap,
                    'mean_gap_hours': max_gap,
                    'outgoing_count': 1,
                    'gap_count': 1
                }
            else:
                # No outgoing communications
                daily_gaps[current_date.isoformat()] = {
                    'max_gap_hours': 24.0,  # Full day
                    'mean_gap_hours': 24.0,
                    'outgoing_count': 0,
                    'gap_count': 0
                }
            
            current_date += timedelta(days=1)
        
        # Calculate weekly statistics
        all_max_gaps = [day_data['max_gap_hours'] for day_data in daily_gaps.values()]
        weekly_max_gap = max(all_max_gaps) if all_max_gaps else 0.0
        mean_daily_gap = sum(all_max_gaps) / len(all_max_gaps) if all_max_gaps else 0.0
        days_with_no_outgoing = sum(1 for day_data in daily_gaps.values() if day_data['outgoing_count'] == 0)
        
        return {
            'weekly_max_gap_hours': weekly_max_gap,
            'mean_daily_gap_hours': mean_daily_gap,
            'days_with_no_outgoing': days_with_no_outgoing,
            'total_outgoing': total_outgoing,
            'daily_gaps': daily_gaps
        }
    
    def _remove_gps_outliers(self, 
                            latitudes: List[float], 
                            longitudes: List[float]) -> Tuple[List[float], List[float]]:
        """Remove outlier GPS points based on distance from center."""
        if not latitudes or not longitudes:
            return latitudes, longitudes
        
        # Calculate center of mass
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
        
        # Calculate distances from center
        distances = []
        for lat, lon in zip(latitudes, longitudes):
            distance = self._haversine_distance(lat, lon, center_lat, center_lon)
            distances.append(distance)
        
        # Calculate statistics
        mean_distance = sum(distances) / len(distances)
        variance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)
        std_distance = math.sqrt(variance)
        
        # Filter outliers
        filtered_latitudes = []
        filtered_longitudes = []
        
        threshold = self.radius_config.outlier_threshold_std * std_distance
        
        for lat, lon, distance in zip(latitudes, longitudes, distances):
            if distance <= mean_distance + threshold:
                filtered_latitudes.append(lat)
                filtered_longitudes.append(lon)
        
        return filtered_latitudes, filtered_longitudes
    
    def _calculate_center_of_mass(self, 
                                 latitudes: List[float], 
                                 longitudes: List[float]) -> Dict[str, Any]:
        """Calculate center of mass of GPS coordinates."""
        if not latitudes or not longitudes:
            return {
                'detected': False,
                'reason': 'No GPS points'
            }
        
        # Simple arithmetic mean (for small distances)
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
        
        return {
            'detected': True,
            'latitude': center_lat,
            'longitude': center_lon,
            'point_count': len(latitudes)
        }
    
    def _calculate_movement_radius(self, 
                                  latitudes: List[float],
                                  longitudes: List[float],
                                  center_of_mass: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate movement radius (radius of gyration)."""
        if not center_of_mass.get('detected', False) or not latitudes or not longitudes:
            return {
                'radius_meters': 0.0,
                'max_distance_meters': 0.0,
                'mean_distance_meters': 0.0,
                'center_detected': False
            }
        
        center_lat = center_of_mass['latitude']
        center_lon = center_of_mass['longitude']
        
        # Calculate distances from center of mass
        distances = []
        for lat, lon in zip(latitudes, longitudes):
            if self.radius_config.use_haversine:
                distance = self._haversine_distance(lat, lon, center_lat, center_lon)
            else:
                # Simple euclidean approximation for small distances
                distance = math.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2) * 111320  # Convert to meters
            
            distances.append(distance)
        
        # Calculate radius of gyration
        mean_distance = sum(distances) / len(distances)
        variance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)
        radius_gyration = math.sqrt(variance)
        
        return {
            'radius_meters': radius_gyration,
            'max_distance_meters': max(distances) if distances else 0.0,
            'mean_distance_meters': mean_distance,
            'center_detected': True,
            'point_count': len(distances)
        }
    
    def _haversine_distance(self, 
                           lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two GPS coordinates.
        
        Args:
            lat1, lon1: First point coordinates in degrees
            lat2, lon2: Second point coordinates in degrees
            
        Returns:
            Distance in meters
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in meters
        earth_radius = 6371000
        
        return earth_radius * c
