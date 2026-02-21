"""
Social Engagement (SE) construct features.

This module implements features related to social engagement patterns,
including communication frequency, contact diversity, and initiation rates.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import math
from collections import defaultdict, Counter


@dataclass
class CommunicationFrequencyConfig:
    """Configuration for communication frequency feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 7  # Weekly analysis
    min_communications_per_day: int = 1  # Minimum to count as active day
    
    # Communication types to include
    include_incoming: bool = False  # Focus on outgoing for engagement
    include_outgoing: bool = True
    communication_types: List[str] = None  # None = include all types
    
    # Data quality requirements
    min_days_with_data: int = 5  # Minimum days with communication data
    min_data_coverage: float = 0.6  # 60% minimum coverage
    
    # Processing options
    exclude_weekends: bool = False  # Optional weekend exclusion
    smooth_daily_counts: bool = True  # Apply smoothing to daily counts


@dataclass
class ContactDiversityConfig:
    """Configuration for contact diversity feature extraction."""
    
    # Analysis parameters
    rolling_window_days: int = 7  # Rolling window for diversity calculation
    analysis_window_days: int = 14  # Total analysis period
    
    # Contact filtering
    min_interactions_per_contact: int = 1  # Minimum interactions to count contact
    exclude_auto_messages: bool = True  # Exclude automated messages
    contact_grouping: bool = True  # Group similar contacts
    
    # Data quality requirements
    min_communications_total: int = 10  # Minimum total communications
    min_days_with_data: int = 5
    
    # Processing options
    normalize_by_frequency: bool = False  # Normalize diversity by communication frequency


@dataclass
class InitiationRateConfig:
    """Configuration for initiation rate feature extraction."""
    
    # Analysis parameters
    analysis_window_days: int = 7  # Weekly analysis
    min_total_communications: int = 5  # Minimum to calculate rate
    
    # Rate calculation options
    handle_zero_division: str = "return_zero"  # "return_zero", "return_nan", "skip"
    include_only_bidirectional: bool = False  # Only count contacts with both directions
    
    # Data quality requirements
    min_days_with_data: int = 5
    min_data_coverage: float = 0.5
    
    # Processing options
    calculate_by_contact: bool = True  # Calculate initiation rate per contact
    exclude_sparse_contacts: bool = True  # Exclude contacts with very few interactions


class SocialEngagementFeatures:
    """
    Implementation of Social Engagement (SE) construct features.
    
    This class provides methods for extracting features related to social
    engagement, which reflects the tendency to initiate and maintain
    social connections and communications.
    
    Attributes:
        freq_config: Configuration for communication frequency features
        diversity_config: Configuration for contact diversity features
        initiation_config: Configuration for initiation rate features
        provenance_tracker: Provenance tracking instance
    """
    
    def __init__(self, 
                 freq_config: Optional[CommunicationFrequencyConfig] = None,
                 diversity_config: Optional[ContactDiversityConfig] = None,
                 initiation_config: Optional[InitiationRateConfig] = None):
        """
        Initialize social engagement features extractor.
        
        Args:
            freq_config: Configuration for communication frequency features
            diversity_config: Configuration for contact diversity features
            initiation_config: Configuration for initiation rate features
        """
        self.freq_config = freq_config or CommunicationFrequencyConfig()
        self.diversity_config = diversity_config or ContactDiversityConfig()
        self.initiation_config = initiation_config or InitiationRateConfig()
        
        # Import provenance tracker locally to avoid circular imports
        try:
            from ..utils.provenance import get_provenance_tracker
            self.provenance_tracker = get_provenance_tracker()
        except ImportError:
            self.provenance_tracker = None
    
    def communication_frequency(self,
                               communication_data: Dict[str, Any],
                               window_start: Optional[datetime] = None,
                               window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate communication frequency from communication logs.
        
        Feature Name: Communication Frequency
        Construct: Social Engagement (SE)
        Mathematical Definition: Count of outgoing communications per day
        Formal Equation: CF = Σ(outgoing_communications_i) for each day i
        Assumptions: Communication data represents meaningful social interactions
        Limitations: Sensitive to platform coverage and communication preferences
        Edge Cases: No communications, single communication days, platform switching
        Output Schema: Daily communication counts with weekly statistics and trends
        
        Args:
            communication_data: Dictionary with 'timestamp', 'direction', 'contact' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing communication frequency values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_communication_frequency",
                input_parameters={
                    "analysis_window_days": self.freq_config.analysis_window_days,
                    "min_communications_per_day": self.freq_config.min_communications_per_day,
                    "include_incoming": self.freq_config.include_incoming,
                    "include_outgoing": self.freq_config.include_outgoing,
                    "exclude_weekends": self.freq_config.exclude_weekends
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
            if len(filtered_timestamps) < self.freq_config.min_communications_per_day:
                raise ValueError(
                    f"Insufficient communication data: {len(filtered_timestamps)} "
                    f"< {self.freq_config.min_communications_per_day}"
                )
            
            # Quality assessment
            quality_metrics = self._assess_communication_quality(
                filtered_timestamps, filtered_directions, filtered_contacts
            )
            
            # Calculate daily communication frequency
            frequency_metrics = self._calculate_communication_frequency(
                filtered_timestamps, filtered_directions, filtered_contacts, 
                window_start, window_end
            )
            
            # Prepare results
            result = {
                'communication_frequency': frequency_metrics,
                'weekly_outgoing_count': frequency_metrics['weekly_outgoing_count'],
                'daily_frequency': frequency_metrics['daily_frequency'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.freq_config.analysis_window_days,
                    'min_communications_per_day': self.freq_config.min_communications_per_day,
                    'include_incoming': self.freq_config.include_incoming,
                    'include_outgoing': self.freq_config.include_outgoing,
                    'exclude_weekends': self.freq_config.exclude_weekends
                },
                'data_summary': {
                    'total_communications': len(filtered_timestamps),
                    'outgoing_communications': frequency_metrics['weekly_outgoing_count'],
                    'incoming_communications': frequency_metrics['weekly_incoming_count'],
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days,
                    'days_with_data': len(frequency_metrics['daily_frequency'])
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'weekly_outgoing_count': frequency_metrics['weekly_outgoing_count'],
                        'days_analyzed': len(frequency_metrics['daily_frequency']),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="communication_frequency",
                    construct="social_engagement",
                    input_data_summary={
                        'communications': len(filtered_timestamps),
                        'outgoing': frequency_metrics['weekly_outgoing_count'],
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400
                    },
                    computation_parameters={
                        'min_communications_per_day': self.freq_config.min_communications_per_day,
                        'include_incoming': self.freq_config.include_incoming
                    },
                    result_summary={
                        'weekly_outgoing_count': frequency_metrics['weekly_outgoing_count'],
                        'mean_daily_frequency': frequency_metrics['mean_daily_frequency'],
                        'active_days': frequency_metrics['active_days']
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
    
    def contact_diversity(self,
                         communication_data: Dict[str, Any],
                         window_start: Optional[datetime] = None,
                         window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate contact diversity from communication logs.
        
        Feature Name: Contact Diversity
        Construct: Social Engagement (SE)
        Mathematical Definition: Unique contacts per rolling 7-day window
        Formal Equation: CD = |{contacts_i}| where contacts_i are contacts in rolling window
        Assumptions: Contact diversity reflects social network breadth and engagement
        Limitations: Sensitive to contact naming conventions and platform differences
        Edge Cases: Single contact, changing contact identifiers, automated messages
        Output Schema: Rolling diversity metrics with contact-level statistics
        
        Args:
            communication_data: Dictionary with 'timestamp', 'direction', 'contact' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing contact diversity values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_contact_diversity",
                input_parameters={
                    "rolling_window_days": self.diversity_config.rolling_window_days,
                    "min_interactions_per_contact": self.diversity_config.min_interactions_per_contact,
                    "exclude_auto_messages": self.diversity_config.exclude_auto_messages,
                    "normalize_by_frequency": self.diversity_config.normalize_by_frequency
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
            if len(filtered_timestamps) < self.diversity_config.min_communications_total:
                raise ValueError(
                    f"Insufficient communication data: {len(filtered_timestamps)} "
                    f"< {self.diversity_config.min_communications_total}"
                )
            
            # Quality assessment
            quality_metrics = self._assess_communication_quality(
                filtered_timestamps, filtered_directions, filtered_contacts
            )
            
            # Calculate contact diversity
            diversity_metrics = self._calculate_contact_diversity(
                filtered_timestamps, filtered_directions, filtered_contacts,
                window_start, window_end
            )
            
            # Prepare results
            result = {
                'contact_diversity': diversity_metrics,
                'weekly_diversity': diversity_metrics['weekly_diversity'],
                'rolling_diversity': diversity_metrics['rolling_diversity'],
                'contact_statistics': diversity_metrics['contact_statistics'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'rolling_window_days': self.diversity_config.rolling_window_days,
                    'min_interactions_per_contact': self.diversity_config.min_interactions_per_contact,
                    'exclude_auto_messages': self.diversity_config.exclude_auto_messages,
                    'normalize_by_frequency': self.diversity_config.normalize_by_frequency
                },
                'data_summary': {
                    'total_communications': len(filtered_timestamps),
                    'unique_contacts': diversity_metrics['weekly_diversity'],
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days,
                    'rolling_windows': len(diversity_metrics['rolling_diversity'])
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'weekly_diversity': diversity_metrics['weekly_diversity'],
                        'rolling_windows': len(diversity_metrics['rolling_diversity']),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="contact_diversity",
                    construct="social_engagement",
                    input_data_summary={
                        'communications': len(filtered_timestamps),
                        'unique_contacts': diversity_metrics['weekly_diversity'],
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400
                    },
                    computation_parameters={
                        'rolling_window_days': self.diversity_config.rolling_window_days,
                        'min_interactions_per_contact': self.diversity_config.min_interactions_per_contact
                    },
                    result_summary={
                        'weekly_diversity': diversity_metrics['weekly_diversity'],
                        'mean_rolling_diversity': diversity_metrics['mean_rolling_diversity'],
                        'diversity_stability': diversity_metrics['diversity_stability']
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
    
    def initiation_rate(self,
                       communication_data: Dict[str, Any],
                       window_start: Optional[datetime] = None,
                       window_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate initiation rate from communication logs.
        
        Feature Name: Initiation Rate
        Construct: Social Engagement (SE)
        Mathematical Definition: IR = Outgoing / (Outgoing + Incoming)
        Formal Equation: IR = Σ(outgoing_i) / Σ(outgoing_i + incoming_i)
        Assumptions: Initiation rate reflects proactive social engagement
        Limitations: Sensitive to communication context and response patterns
        Edge Cases: Zero total communications, single-direction communications
        Output Schema: Initiation rate with contact-level analysis and quality metrics
        
        Args:
            communication_data: Dictionary with 'timestamp', 'direction', 'contact' columns
            window_start: Start of analysis window. If None, uses data start.
            window_end: End of analysis window. If None, uses data end.
            
        Returns:
            Dictionary containing initiation rate values and metadata
            
        Raises:
            ValueError: If required data columns are missing or insufficient data
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="extract_initiation_rate",
                input_parameters={
                    "analysis_window_days": self.initiation_config.analysis_window_days,
                    "min_total_communications": self.initiation_config.min_total_communications,
                    "handle_zero_division": self.initiation_config.handle_zero_division,
                    "include_only_bidirectional": self.initiation_config.include_only_bidirectional
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
            if len(filtered_timestamps) < self.initiation_config.min_total_communications:
                raise ValueError(
                    f"Insufficient communication data: {len(filtered_timestamps)} "
                    f"< {self.initiation_config.min_total_communications}"
                )
            
            # Quality assessment
            quality_metrics = self._assess_communication_quality(
                filtered_timestamps, filtered_directions, filtered_contacts
            )
            
            # Calculate initiation rate
            initiation_metrics = self._calculate_initiation_rate(
                filtered_timestamps, filtered_directions, filtered_contacts
            )
            
            # Prepare results
            result = {
                'initiation_rate': initiation_metrics,
                'weekly_initiation_rate': initiation_metrics['weekly_initiation_rate'],
                'contact_initiation_rates': initiation_metrics['contact_initiation_rates'],
                'direction_counts': initiation_metrics['direction_counts'],
                'quality_metrics': quality_metrics,
                'processing_parameters': {
                    'analysis_window_days': self.initiation_config.analysis_window_days,
                    'min_total_communications': self.initiation_config.min_total_communications,
                    'handle_zero_division': self.initiation_config.handle_zero_division,
                    'include_only_bidirectional': self.initiation_config.include_only_bidirectional
                },
                'data_summary': {
                    'total_communications': len(filtered_timestamps),
                    'outgoing_count': initiation_metrics['direction_counts']['outgoing'],
                    'incoming_count': initiation_metrics['direction_counts']['incoming'],
                    'analysis_start': window_start.isoformat(),
                    'analysis_end': window_end.isoformat(),
                    'date_range_days': (window_end - window_start).days,
                    'unique_contacts': len(initiation_metrics['contact_initiation_rates'])
                }
            }
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'weekly_initiation_rate': initiation_metrics['weekly_initiation_rate'],
                        'contacts_analyzed': len(initiation_metrics['contact_initiation_rates']),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record feature extraction provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name="initiation_rate",
                    construct="social_engagement",
                    input_data_summary={
                        'communications': len(filtered_timestamps),
                        'outgoing': initiation_metrics['direction_counts']['outgoing'],
                        'incoming': initiation_metrics['direction_counts']['incoming'],
                        'time_span_days': (max(filtered_timestamps) - min(filtered_timestamps)).total_seconds() / 86400
                    },
                    computation_parameters={
                        'handle_zero_division': self.initiation_config.handle_zero_division,
                        'include_only_bidirectional': self.initiation_config.include_only_bidirectional
                    },
                    result_summary={
                        'weekly_initiation_rate': initiation_metrics['weekly_initiation_rate'],
                        'direction_balance': initiation_metrics['direction_balance'],
                        'proactive_contacts': initiation_metrics['proactive_contacts']
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
                'contact_diversity': 0.0,
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
        
        # Calculate contact diversity
        unique_contacts = len(set(contacts))
        contact_diversity = unique_contacts / len(contacts) if contacts else 0
        
        # Calculate coverage ratio
        expected_communications = time_span_days * 10  # 10 communications/day as reference
        coverage_ratio = min(len(timestamps) / expected_communications, 1.0) if expected_communications > 0 else 0
        
        # Overall quality score
        quality_score = (
            coverage_ratio * 0.3 +
            min(1.0, communications_per_day / 20) * 0.3 +  # 20 communications/day as reference
            direction_balance * 0.2 +
            min(1.0, contact_diversity * 5) * 0.2  # Normalize contact diversity
        )
        
        return {
            'coverage_ratio': coverage_ratio,
            'communications_per_day': communications_per_day,
            'direction_balance': direction_balance,
            'contact_diversity': contact_diversity,
            'overall_quality': quality_score
        }
    
    def _calculate_communication_frequency(self, 
                                          timestamps: List[datetime],
                                          directions: List[str],
                                          contacts: List[str],
                                          window_start: datetime,
                                          window_end: datetime) -> Dict[str, Any]:
        """Calculate daily communication frequency."""
        # Group communications by day
        daily_counts = defaultdict(int)
        daily_outgoing = defaultdict(int)
        daily_incoming = defaultdict(int)
        
        for ts, direction in zip(timestamps, directions):
            date_key = ts.date().isoformat()
            daily_counts[date_key] += 1
            
            if direction.lower() in ['outgoing', 'out']:
                if self.freq_config.include_outgoing:
                    daily_outgoing[date_key] += 1
            elif direction.lower() in ['incoming', 'in']:
                if self.freq_config.include_incoming:
                    daily_incoming[date_key] += 1
        
        # Apply weekend exclusion if configured
        if self.freq_config.exclude_weekends:
            filtered_daily_counts = {}
            filtered_daily_outgoing = {}
            filtered_daily_incoming = {}
            
            for date_str, count in daily_counts.items():
                date = datetime.fromisoformat(date_str)
                if date.weekday() < 5:  # Monday-Friday
                    filtered_daily_counts[date_str] = count
                    filtered_daily_outgoing[date_str] = daily_outgoing[date_str]
                    filtered_daily_incoming[date_str] = daily_incoming[date_str]
            
            daily_counts = filtered_daily_counts
            daily_outgoing = filtered_daily_outgoing
            daily_incoming = filtered_daily_incoming
        
        # Apply smoothing if configured
        if self.freq_config.smooth_daily_counts and len(daily_counts) > 2:
            daily_outgoing = self._smooth_daily_counts(daily_outgoing)
        
        # Calculate statistics
        total_outgoing = sum(daily_outgoing.values())
        total_incoming = sum(daily_incoming.values())
        active_days = len([count for count in daily_outgoing.values() if count >= self.freq_config.min_communications_per_day])
        
        mean_daily_frequency = total_outgoing / len(daily_outgoing) if daily_outgoing else 0.0
        
        # Prepare daily frequency data
        daily_frequency = {}
        for date_str in daily_counts:
            daily_frequency[date_str] = {
                'total_communications': daily_counts[date_str],
                'outgoing_communications': daily_outgoing.get(date_str, 0),
                'incoming_communications': daily_incoming.get(date_str, 0),
                'is_active': daily_outgoing.get(date_str, 0) >= self.freq_config.min_communications_per_day
            }
        
        return {
            'weekly_outgoing_count': total_outgoing,
            'weekly_incoming_count': total_incoming,
            'mean_daily_frequency': mean_daily_frequency,
            'active_days': active_days,
            'daily_frequency': daily_frequency
        }
    
    def _smooth_daily_counts(self, daily_counts: Dict[str, int]) -> Dict[str, int]:
        """Apply moving average smoothing to daily counts."""
        if not daily_counts:
            return daily_counts
        
        # Sort dates
        sorted_dates = sorted(daily_counts.keys())
        
        # Apply 3-day moving average
        smoothed_counts = {}
        for i, date in enumerate(sorted_dates):
            if i == 0:
                # First day - average with next day
                next_date = sorted_dates[1] if len(sorted_dates) > 1 else date
                smoothed_counts[date] = (daily_counts[date] + daily_counts[next_date]) / 2
            elif i == len(sorted_dates) - 1:
                # Last day - average with previous day
                prev_date = sorted_dates[i-1]
                smoothed_counts[date] = (daily_counts[date] + daily_counts[prev_date]) / 2
            else:
                # Middle days - 3-day average
                prev_date = sorted_dates[i-1]
                next_date = sorted_dates[i+1]
                smoothed_counts[date] = (daily_counts[prev_date] + daily_counts[date] + daily_counts[next_date]) / 3
        
        return smoothed_counts
    
    def _calculate_contact_diversity(self, 
                                   timestamps: List[datetime],
                                   directions: List[str],
                                   contacts: List[str],
                                   window_start: datetime,
                                   window_end: datetime) -> Dict[str, Any]:
        """Calculate rolling contact diversity."""
        # Filter contacts by minimum interactions
        contact_counts = Counter(contacts)
        valid_contacts = {contact for contact, count in contact_counts.items() 
                         if count >= self.diversity_config.min_interactions_per_contact}
        
        # Exclude automated messages if configured
        if self.diversity_config.exclude_auto_messages:
            auto_patterns = ['system', 'auto', 'bot', 'notification', 'alert']
            valid_contacts = {contact for contact in valid_contacts 
                            if not any(pattern in contact.lower() for pattern in auto_patterns)}
        
        # Group communications by day with valid contacts
        daily_contacts = defaultdict(set)
        for ts, contact in zip(timestamps, contacts):
            if contact in valid_contacts:
                date_key = ts.date().isoformat()
                daily_contacts[date_key].add(contact)
        
        # Calculate rolling diversity
        rolling_diversity = {}
        sorted_dates = sorted(daily_contacts.keys())
        
        for i, date in enumerate(sorted_dates):
            # Get rolling window
            window_start_date = datetime.fromisoformat(date) - timedelta(days=self.diversity_config.rolling_window_days - 1)
            window_end_date = datetime.fromisoformat(date)
            
            # Collect contacts in rolling window
            window_contacts = set()
            for window_date in sorted_dates:
                current_date = datetime.fromisoformat(window_date)
                if window_start_date <= current_date <= window_end_date:
                    window_contacts.update(daily_contacts[window_date])
            
            rolling_diversity[date] = len(window_contacts)
        
        # Calculate weekly diversity
        all_contacts = set()
        for contacts_set in daily_contacts.values():
            all_contacts.update(contacts_set)
        
        weekly_diversity = len(all_contacts)
        
        # Calculate statistics
        if rolling_diversity:
            mean_rolling_diversity = sum(rolling_diversity.values()) / len(rolling_diversity)
            diversity_values = list(rolling_diversity.values())
            diversity_stability = 1.0 - (max(diversity_values) - min(diversity_values)) / mean_rolling_diversity if mean_rolling_diversity > 0 else 0
        else:
            mean_rolling_diversity = 0.0
            diversity_stability = 0.0
        
        # Contact-level statistics
        contact_statistics = {}
        for contact in valid_contacts:
            contact_communications = [i for i, c in enumerate(contacts) if c == contact]
            contact_statistics[contact] = {
                'total_communications': len(contact_communications),
                'first_contact': timestamps[contact_communications[0]].isoformat(),
                'last_contact': timestamps[contact_communications[-1]].isoformat(),
                'communication_span_days': (timestamps[contact_communications[-1]] - timestamps[contact_communications[0]]).days + 1
            }
        
        return {
            'weekly_diversity': weekly_diversity,
            'rolling_diversity': rolling_diversity,
            'mean_rolling_diversity': mean_rolling_diversity,
            'diversity_stability': diversity_stability,
            'contact_statistics': contact_statistics
        }
    
    def _calculate_initiation_rate(self, 
                                 timestamps: List[datetime],
                                 directions: List[str],
                                 contacts: List[str]) -> Dict[str, Any]:
        """Calculate initiation rate metrics."""
        # Count directions
        outgoing_count = sum(1 for d in directions if d.lower() in ['outgoing', 'out'])
        incoming_count = sum(1 for d in directions if d.lower() in ['incoming', 'in'])
        total_count = outgoing_count + incoming_count
        
        # Calculate overall initiation rate
        if total_count == 0:
            if self.initiation_config.handle_zero_division == "return_zero":
                weekly_initiation_rate = 0.0
            elif self.initiation_config.handle_zero_division == "return_nan":
                weekly_initiation_rate = float('nan')
            else:  # "skip"
                weekly_initiation_rate = None
        else:
            weekly_initiation_rate = outgoing_count / total_count
        
        # Calculate per-contact initiation rates
        contact_directions = defaultdict(lambda: {'outgoing': 0, 'incoming': 0})
        for direction, contact in zip(directions, contacts):
            if direction.lower() in ['outgoing', 'out']:
                contact_directions[contact]['outgoing'] += 1
            elif direction.lower() in ['incoming', 'in']:
                contact_directions[contact]['incoming'] += 1
        
        contact_initiation_rates = {}
        for contact, counts in contact_directions.items():
            contact_total = counts['outgoing'] + counts['incoming']
            
            # Apply bidirectional filter if configured
            if self.initiation_config.include_only_bidirectional:
                if counts['outgoing'] == 0 or counts['incoming'] == 0:
                    continue
            
            # Apply sparse contact filter if configured
            if self.initiation_config.exclude_sparse_contacts and contact_total < 3:
                continue
            
            if contact_total == 0:
                contact_rate = 0.0
            else:
                contact_rate = counts['outgoing'] / contact_total
            
            contact_initiation_rates[contact] = {
                'initiation_rate': contact_rate,
                'outgoing_count': counts['outgoing'],
                'incoming_count': counts['incoming'],
                'total_count': contact_total
            }
        
        # Calculate additional metrics
        direction_balance = 1.0 - abs(outgoing_count - incoming_count) / total_count if total_count > 0 else 0
        proactive_contacts = len([c for c, data in contact_initiation_rates.items() if data['initiation_rate'] > 0.5])
        
        return {
            'weekly_initiation_rate': weekly_initiation_rate,
            'contact_initiation_rates': contact_initiation_rates,
            'direction_counts': {
                'outgoing': outgoing_count,
                'incoming': incoming_count,
                'total': total_count
            },
            'direction_balance': direction_balance,
            'proactive_contacts': proactive_contacts,
            'total_contacts_analyzed': len(contact_initiation_rates)
        }
