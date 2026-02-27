"""
Construct Aggregator Module for Psyconstruct.

This module aggregates individual features into construct-level scores using
various normalization and aggregation strategies with comprehensive provenance tracking.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import math
import statistics
from pathlib import Path

from ..features import (
    BehavioralActivationFeatures,
    AvoidanceFeatures,
    SocialEngagementFeatures,
    RoutineStabilityFeatures
)
from .registry import ConstructRegistry


@dataclass
class AggregationConfig:
    """Configuration for construct aggregation."""
    
    # Normalization options
    normalization_method: str = "zscore"  # "zscore", "minmax", "robust", "none"
    within_participant: bool = True  # Normalize within participant or across population
    reference_population: Optional[str] = None  # Reference population for normalization
    
    # Aggregation options
    aggregation_method: str = "weighted_mean"  # "weighted_mean", "unweighted_mean", "median"
    handle_missing: str = "exclude"  # "exclude", "mean_impute", "median_impute", "zero_impute"
    
    # Quality requirements
    min_features_required: int = 2  # Minimum features required for aggregation
    min_quality_threshold: float = 0.5  # Minimum quality threshold
    
    # Output options
    include_feature_scores: bool = True  # Include individual feature scores in output
    include_quality_metrics: bool = True  # Include quality metrics in output
    include_normalization_params: bool = True  # Include normalization parameters


@dataclass
class ConstructScore:
    """Dataclass for construct-level scores."""
    
    construct_name: str
    score: float
    normalized_score: float
    feature_scores: Dict[str, float]
    quality_metrics: Dict[str, Any]
    aggregation_parameters: Dict[str, Any]
    timestamp: datetime
    participant_id: Optional[str] = None
    dispersion_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None


class ConstructAggregator:
    """
    Aggregates individual features into construct-level scores.
    
    This class provides methods for normalizing features, aggregating them
    according to construct definitions, and generating comprehensive
    construct-level scores with quality metrics and provenance tracking.
    
    MEASUREMENT MODEL NOTE:
    The registry defines reflective vs formative measurement models, but
    this v1.0 implementation uses linear weighted aggregation for ALL constructs.
    Measurement models are currently descriptive only - they are not operationalized
    in the aggregation logic. All constructs are treated as composites with weighted
    linear combination of features.
    
    Future versions should operationalize measurement models to apply different
    aggregation strategies for reflective vs formative constructs.
    
    Attributes:
        config: Configuration for aggregation behavior
        construct_registry: Registry of construct definitions and weights
        provenance_tracker: Provenance tracking instance
    """
    
    def __init__(self, 
                 config: Optional[AggregationConfig] = None,
                 construct_registry: Optional[ConstructRegistry] = None):
        """
        Initialize construct aggregator.
        
        Args:
            config: Configuration for aggregation behavior
            construct_registry: ConstructRegistry instance (if None, creates default)
        """
        self.config = config or AggregationConfig()
        
        # Use provided registry or create default
        self.construct_registry = construct_registry or ConstructRegistry()
        
        # Initialize provenance tracker
        try:
            from ..utils.provenance import get_provenance_tracker
            self.provenance_tracker = get_provenance_tracker()
        except ImportError:
            self.provenance_tracker = None
    
    def aggregate_construct(self,
                           construct_name: str,
                           feature_results: Dict[str, Any],
                           participant_id: Optional[str] = None,
                           reference_data: Optional[Dict[str, List[float]]] = None) -> ConstructScore:
        """
        Aggregate features into a construct-level score.
        
        Args:
            construct_name: Name of the construct to aggregate
            feature_results: Dictionary of feature extraction results
            participant_id: Optional participant identifier
            reference_data: Optional reference data for normalization
            
        Returns:
            ConstructScore object with aggregated results
            
        Raises:
            ValueError: If construct is not found or insufficient features
        """
        # Start provenance tracking
        operation_id = None
        if self.provenance_tracker:
            operation_id = self.provenance_tracker.start_operation(
                operation_type="aggregate_construct",
                input_parameters={
                    "construct_name": construct_name,
                    "normalization_method": self.config.normalization_method,
                    "aggregation_method": self.config.aggregation_method,
                    "participant_id": participant_id
                }
            )
        
        try:
            # Validate construct exists
            construct_def = self.construct_registry.get_construct(construct_name)
            
            # Extract feature values and quality
            feature_values = {}
            feature_qualities = {}
            feature_weights = {}
            feature_directions = {}
            feature_aggregations = {}
            
            for feature_def in construct_def.features:
                feature_name = feature_def.name
                
                if feature_name in feature_results:
                    feature_result = feature_results[feature_name]
                    
                    # Extract primary value from feature result
                    primary_value = self._extract_primary_value(feature_result, feature_name)
                    if primary_value is not None:
                        feature_values[feature_name] = primary_value
                        feature_qualities[feature_name] = self._extract_quality_score(feature_result)
                        feature_weights[feature_name] = feature_def.weight
                        feature_directions[feature_name] = feature_def.direction
                        feature_aggregations[feature_name] = feature_def.aggregation
            
            # Check minimum features requirement
            if len(feature_values) < self.config.min_features_required:
                raise ValueError(
                    f"Insufficient features for {construct_name}: "
                    f"{len(feature_values)} < {self.config.min_features_required}"
                )
            
            # Filter by quality threshold
            high_quality_features = {
                name: value for name, value in feature_values.items()
                if feature_qualities.get(name, 0) >= self.config.min_quality_threshold
            }
            
            if len(high_quality_features) < self.config.min_features_required:
                raise ValueError(
                    f"Insufficient high-quality features for {construct_name}: "
                    f"{len(high_quality_features)} < {self.config.min_features_required}"
                )
            
            # Apply directional transforms before normalization
            directionally_transformed = self._apply_directional_transforms(
                high_quality_features, feature_directions, feature_aggregations
            )
            
            # Normalize features
            normalized_features = self._normalize_features(
                directionally_transformed, construct_name, reference_data
            )
            
            # Aggregate into construct score using registry-defined aggregation
            raw_score = self._aggregate_features_registry(
                high_quality_features, feature_weights, feature_aggregations, construct_def
            )
            
            normalized_score = self._aggregate_features_registry(
                normalized_features, feature_weights, feature_aggregations, construct_def
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_aggregation_quality(
                feature_qualities, high_quality_features
            )
            
            # Calculate dispersion interval
            dispersion_interval = self._calculate_dispersion_interval(
                normalized_features, feature_weights
            )
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                construct_name, normalized_score, quality_metrics
            )
            
            # Prepare aggregation parameters
            aggregation_parameters = {
                "normalization_method": self.config.normalization_method,
                "aggregation_method": self.config.aggregation_method,
                "features_used": list(high_quality_features.keys()),
                "feature_weights": {name: feature_weights[name] for name in high_quality_features},
                "quality_threshold": self.config.min_quality_threshold
            }
            
            # Create construct score
            construct_score = ConstructScore(
                construct_name=construct_name,
                score=raw_score,
                normalized_score=normalized_score,
                feature_scores=normalized_features if self.config.include_feature_scores else {},
                quality_metrics=quality_metrics,
                aggregation_parameters=aggregation_parameters,
                timestamp=datetime.now(),
                participant_id=participant_id,
                dispersion_interval=dispersion_interval,
                interpretation=interpretation
            )
            
            # Record provenance
            if self.provenance_tracker:
                self.provenance_tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary={
                        'success': True,
                        'construct_score': normalized_score,
                        'features_used': len(high_quality_features),
                        'quality_score': quality_metrics['overall_quality']
                    },
                    duration_seconds=0.0
                )
                
                # Record aggregation provenance
                self.provenance_tracker.record_feature_extraction(
                    feature_name=f"{construct_name}_aggregate",
                    construct=construct_name,
                    input_data_summary={
                        'features': len(feature_results),
                        'high_quality_features': len(high_quality_features)
                    },
                    computation_parameters=aggregation_parameters,
                    result_summary={
                        'raw_score': raw_score,
                        'normalized_score': normalized_score,
                        'dispersion_interval': dispersion_interval
                    },
                    data_quality_metrics=quality_metrics,
                    algorithm_version="1.0.0"
                )
            
            return construct_score
            
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
    
    def aggregate_all_constructs(self,
                                feature_results: Dict[str, Any],
                                participant_id: Optional[str] = None,
                                reference_data: Optional[Dict[str, Dict[str, List[float]]]] = None) -> Dict[str, ConstructScore]:
        """
        Aggregate all available constructs.
        
        Args:
            feature_results: Dictionary of all feature extraction results
            participant_id: Optional participant identifier
            reference_data: Optional reference data for each construct
            
        Returns:
            Dictionary mapping construct names to ConstructScore objects
        """
        construct_scores = {}
        
        for construct_name in self.construct_registry.constructs:
            try:
                construct_reference = None
                if reference_data and construct_name in reference_data:
                    construct_reference = reference_data[construct_name]
                
                score = self.aggregate_construct(
                    construct_name, feature_results, participant_id, construct_reference
                )
                construct_scores[construct_name] = score
                
            except ValueError as e:
                # Log failed aggregation but continue with others
                if self.provenance_tracker:
                    print(f"Warning: Failed to aggregate {construct_name}: {str(e)}")
                continue
        
        return construct_scores
    
    def _extract_primary_value(self, feature_result: Dict[str, Any], feature_name: str) -> Optional[float]:
        """Extract primary value from feature result."""
        # Define primary value extraction for each feature type
        primary_value_mappings = {
            # Behavioral Activation features
            "activity_volume": lambda r: r.get("weekly_activity_count"),
            "location_diversity": lambda r: r.get("shannon_entropy"),
            "app_usage_breadth": lambda r: r.get("daily_breadth"),
            "activity_timing_variance": lambda r: r.get("timing_variance"),
            
            # Avoidance features
            "home_confinement": lambda r: r.get("home_confinement_percentage"),
            "communication_gaps": lambda r: r.get("max_daily_gap_hours"),
            "movement_radius": lambda r: r.get("radius_of_gyration_meters"),
            
            # Social Engagement features
            "communication_frequency": lambda r: r.get("weekly_outgoing_count"),
            "contact_diversity": lambda r: r.get("weekly_diversity"),
            "initiation_rate": lambda r: r.get("weekly_initiation_rate"),
            
            # Routine Stability features
            "sleep_onset_consistency": lambda r: r.get("sleep_onset_sd_hours"),
            "sleep_duration": lambda r: r.get("mean_sleep_duration_hours"),
            "activity_fragmentation": lambda r: r.get("mean_entropy"),
            "circadian_midpoint": lambda r: r.get("mean_midpoint_hour")
        }
        
        if feature_name in primary_value_mappings:
            value = primary_value_mappings[feature_name](feature_result)
            return float(value) if value is not None else None
        
        # Default extraction - try common keys
        for key in ["score", "value", "mean", "count", "percentage"]:
            if key in feature_result:
                value = feature_result[key]
                if isinstance(value, (int, float)):
                    return float(value)
        
        return None
    
    def _extract_quality_score(self, feature_result: Dict[str, Any]) -> float:
        """Extract quality score from feature result."""
        # Try quality_metrics first
        if "quality_metrics" in feature_result:
            quality_metrics = feature_result["quality_metrics"]
            if "overall_quality" in quality_metrics:
                return float(quality_metrics["overall_quality"])
        
        # Try direct quality score
        for key in ["quality", "quality_score", "data_quality"]:
            if key in feature_result:
                value = feature_result[key]
                if isinstance(value, (int, float)):
                    return float(value)
        
        # Default quality score
        return 1.0
    
    def _normalize_features(self,
                           feature_values: Dict[str, float],
                           construct_name: str,
                           reference_data: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
        """Normalize feature values."""
        if self.config.normalization_method == "none":
            return feature_values
        
        normalized = {}
        
        for feature_name, value in feature_values.items():
            if self.config.normalization_method == "zscore":
                normalized[feature_name] = self._zscore_normalize(
                    value, feature_name, reference_data
                )
            elif self.config.normalization_method == "minmax":
                normalized[feature_name] = self._minmax_normalize(
                    value, feature_name, reference_data
                )
            elif self.config.normalization_method == "robust":
                normalized[feature_name] = self._robust_normalize(
                    value, feature_name, reference_data
                )
        
        return normalized
    
    def _zscore_normalize(self,
                         value: float,
                         feature_name: str,
                         reference_data: Optional[Dict[str, List[float]]] = None) -> float:
        """Z-score normalization."""
        if reference_data and feature_name in reference_data:
            ref_values = reference_data[feature_name]
            mean = statistics.mean(ref_values)
            std = statistics.stdev(ref_values) if len(ref_values) > 1 else 1.0
        else:
            # If no reference data available, raise error for explicit z-score request
            if self.config.normalization_method == "zscore":
                raise ValueError(
                    f"Reference data required for z-score normalization of '{feature_name}'. "
                    f"Either provide reference_data or use normalization_method='none'"
                )
            # Fallback to identity transformation
            mean = 0.0
            std = 1.0
        
        return (value - mean) / std if std != 0 else 0.0
    
    def _minmax_normalize(self,
                          value: float,
                          feature_name: str,
                          reference_data: Optional[Dict[str, List[float]]] = None) -> float:
        """Min-max normalization."""
        if reference_data and feature_name in reference_data:
            ref_values = reference_data[feature_name]
            min_val = min(ref_values)
            max_val = max(ref_values)
        else:
            # If no reference data available, raise error for explicit minmax request
            if self.config.normalization_method == "minmax":
                raise ValueError(
                    f"Reference data required for min-max normalization of '{feature_name}'. "
                    f"Either provide reference_data or use normalization_method='none'"
                )
            # Fallback to identity transformation
            min_val = 0.0
            max_val = 1.0
        
        range_val = max_val - min_val
        return (value - min_val) / range_val if range_val != 0 else 0.0
    
    def _robust_normalize(self,
                          value: float,
                          feature_name: str,
                          reference_data: Optional[Dict[str, List[float]]] = None) -> float:
        """Robust normalization using median and MAD."""
        if reference_data and feature_name in reference_data:
            ref_values = reference_data[feature_name]
            median = statistics.median(ref_values)
            # Median absolute deviation
            mad = statistics.median([abs(x - median) for x in ref_values])
        else:
            median = 0.0
            mad = 1.0
        
        return (value - median) / mad if mad != 0 else 0.0
    
    def _aggregate_features_registry(self,
                                       feature_values: Dict[str, float],
                                       feature_weights: Dict[str, float],
                                       feature_aggregations: Dict[str, str],
                                       construct_def) -> float:
        """
        Aggregate features using registry-defined methods and measurement models.
        
        Operationalizes reflective vs formative measurement models:
        - Reflective: Weighted average with internal consistency considerations
        - Formative: Weighted sum without internal consistency assumptions
        
        Args:
            feature_values: Normalized feature values
            feature_weights: Feature weights
            feature_aggregations: Aggregation methods for each feature
            construct_def: Construct definition with measurement model
            
        Returns:
            Aggregated construct score
        """
        measurement_model = construct_def.measurement_model.lower()
        
        if measurement_model == "reflective":
            return self._aggregate_reflective(
                feature_values, feature_weights, feature_aggregations, construct_def
            )
        elif measurement_model == "formative":
            return self._aggregate_formative(
                feature_values, feature_weights, feature_aggregations, construct_def
            )
        else:
            # Default to composite aggregation for unspecified models
            return self._aggregate_composite(
                feature_values, feature_weights, feature_aggregations
            )
    
    def _aggregate_reflective(self,
                             feature_values: Dict[str, float],
                             feature_weights: Dict[str, float],
                             feature_aggregations: Dict[str, str],
                             construct_def) -> float:
        """
        Aggregate features using reflective measurement model.
        
        In reflective models, indicators are caused by the latent construct.
        Features should be internally consistent and interchangeable.
        """
        # Apply measurement model specific transformations
        transformed_values = {}
        for feature_name, value in feature_values.items():
            aggregation_method = feature_aggregations.get(feature_name, "mean")
            
            if aggregation_method == "circular_sd":
                # For reflective models, circular SD represents dispersion
                # Lower values indicate stronger construct manifestation
                transformed_values[feature_name] = -abs(value)  # Invert for consistency
            elif aggregation_method in ["directional_inverse", "directional_reverse"]:
                # Directional transforms already applied in preprocessing
                transformed_values[feature_name] = value
            else:
                transformed_values[feature_name] = value
        
        # Calculate weighted average (reflective models assume interchangeability)
        weighted_sum = sum(
            transformed_values[feature] * feature_weights[feature]
            for feature in transformed_values
        )
        total_weight = sum(feature_weights[feature] for feature in transformed_values)
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _aggregate_formative(self,
                            feature_values: Dict[str, float],
                            feature_weights: Dict[str, float],
                            feature_aggregations: Dict[str, str],
                            construct_def) -> float:
        """
        Aggregate features using formative measurement model.
        
        In formative models, indicators cause the latent construct.
        Features are not assumed to be interchangeable or internally consistent.
        """
        # Apply measurement model specific transformations
        transformed_values = {}
        for feature_name, value in feature_values.items():
            aggregation_method = feature_aggregations.get(feature_name, "mean")
            
            if aggregation_method == "circular_sd":
                # For formative models, circular SD is a distinct dimension
                transformed_values[feature_name] = value
            elif aggregation_method in ["directional_inverse", "directional_reverse"]:
                # Directional transforms already applied in preprocessing
                transformed_values[feature_name] = value
            else:
                transformed_values[feature_name] = value
        
        # Calculate weighted sum (formative models do not assume interchangeability)
        weighted_sum = sum(
            transformed_values[feature] * feature_weights[feature]
            for feature in transformed_values
        )
        
        # No normalization by total weight for formative models
        # Each feature contributes according to its specified weight
        return weighted_sum
    
    def _aggregate_composite(self,
                            feature_values: Dict[str, float],
                            feature_weights: Dict[str, float],
                            feature_aggregations: Dict[str, str]) -> float:
        """
        Aggregate features using default composite approach.
        
        Used when measurement model is not specified or for backward compatibility.
        """
        # Default weighted average approach
        weighted_sum = sum(
            feature_values[feature] * feature_weights[feature]
            for feature in feature_values
        )
        total_weight = sum(feature_weights[feature] for feature in feature_values)
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _apply_directional_transforms(self,
                                     feature_values: Dict[str, float],
                                     feature_directions: Dict[str, Optional[str]],
                                     feature_aggregations: Dict[str, str]) -> Dict[str, float]:
        """
        Apply directional transforms to features before normalization.
        
        Implements registry formulas with epsilon stabilization:
        - directional_inverse: "standardize(-x)" with epsilon stabilization
        - directional_reverse: "standardize(max(x) - x)" with epsilon stabilization
        
        Args:
            feature_values: Raw feature values
            feature_directions: Direction for each feature ("positive", "negative", or None)
            feature_aggregations: Aggregation method for each feature
            
        Returns:
            Transformed feature values
        """
        transformed = {}
        epsilon = 1e-8  # Small constant for numerical stability
        
        for feature_name, value in feature_values.items():
            direction = feature_directions.get(feature_name)
            aggregation = feature_aggregations.get(feature_name, "mean")
            
            if aggregation == "directional_inverse" and direction == "negative":
                # Registry formula: "standardize(-x)" with epsilon stabilization
                # Apply stabilized negation to prevent exploding values
                transformed[feature_name] = -value / (abs(value) + epsilon)
                
            elif aggregation == "directional_reverse" and direction == "positive":
                # Registry formula: "standardize(max(x) - x)" 
                # For consistency measures where lower values indicate higher construct
                # Get feature range from registry for proper reversal
                construct_def = self.construct_registry.get_construct_for_feature(feature_name)
                if construct_def:
                    feature_def = construct_def.get_feature(feature_name)
                    if feature_def and feature_def.expected_range:
                        min_val, max_val = feature_def.expected_range
                        if max_val is not None and min_val is not None:
                            # Proper directional reversal with epsilon stabilization
                            range_val = max_val - min_val
                            if range_val > epsilon:
                                transformed[feature_name] = (max_val - value) / range_val
                            else:
                                transformed[feature_name] = -value / (abs(value) + epsilon)
                        else:
                            # Fallback to stabilized negation if range undefined
                            transformed[feature_name] = -value / (abs(value) + epsilon)
                    else:
                        # Fallback for undefined feature
                        transformed[feature_name] = -value / (abs(value) + epsilon)
                else:
                    # Ultimate fallback
                    transformed[feature_name] = -value / (abs(value) + epsilon)
                
            elif aggregation == "circular_sd" and feature_name == "circadian_midpoint":
                # Apply circular transformation for time-based features
                transformed[feature_name] = self._apply_circular_transform(value)
                
            else:
                # No transformation needed
                transformed[feature_name] = value
        
        return transformed
    
    def _calculate_aggregation_quality(self,
                                      feature_qualities: Dict[str, float],
                                      used_features: Dict[str, float]) -> Dict[str, Any]:
        """Calculate quality metrics for aggregation."""
        used_qualities = {
            name: feature_qualities.get(name, 1.0) 
            for name in used_features
        }
        
        # Weighted average quality
        weights = list(used_features.values())
        qualities = list(used_qualities.values())
        
        if weights and qualities:
            weighted_quality = sum(w * q for w, q in zip(weights, qualities)) / sum(weights)
        else:
            weighted_quality = 0.0
        
        # Quality consistency (lower variance = more consistent)
        if len(qualities) > 1:
            quality_variance = statistics.variance(qualities)
            quality_consistency = 1.0 / (1.0 + quality_variance)  # Higher for lower variance
        else:
            quality_variance = 0.0
            quality_consistency = 1.0
        
        return {
            'overall_quality': weighted_quality,
            'quality_variance': quality_variance,
            'quality_consistency': quality_consistency,
            'feature_count': len(used_features),
            'feature_qualities': used_qualities
        }
    
    def calculate_construct_reliability(self,
                                       construct_name: str,
                                       feature_values: Dict[str, float],
                                       construct_def) -> Dict[str, float]:
        """
        Calculate construct-level reliability estimates.
        
        NOTE: True reliability estimation requires multi-participant data to calculate
        item covariances and inter-item correlations. This single-participant
        implementation cannot compute psychometrically valid reliability coefficients.
        
        Args:
            construct_name: Name of the construct
            feature_values: Feature values for this participant
            construct_def: Construct definition with measurement model
            
        Returns:
            Dictionary with reliability estimation limitations noted
        """
        measurement_model = construct_def.measurement_model.lower()
        features = list(feature_values.keys())
        
        if len(features) < 2:
            return {
                'cronbachs_alpha': None,
                'composite_reliability': None,
                'measurement_model': measurement_model,
                'reliability_type': 'insufficient_features',
                'message': 'Need at least 2 features for reliability estimation'
            }
        
        # Return limitation notice for all reliability calculations
        return {
            'cronbachs_alpha': None,
            'composite_reliability': None,
            'measurement_model': measurement_model,
            'reliability_type': 'not_computable',
            'message': 'Reliability estimation requires multi-participant data for item covariances. Single-participant reliability not psychometrically valid.'
        }
    
    def _calculate_cronbachs_alpha(self,
                                  construct_name: str,
                                  feature_values: Dict[str, float],
                                  construct_def) -> Dict[str, float]:
        """
        Placeholder for Cronbach's alpha calculation.
        
        NOTE: This method is disabled because Cronbach's alpha requires
        multi-participant data to calculate item covariances. Single-participant
        reliability estimation is not psychometrically valid.
        
        This method is retained for API compatibility but returns None.
        """
        return {
            'cronbachs_alpha': None,
            'reliability_type': 'not_computable',
            'message': 'Cronbach\'s alpha requires multi-participant covariance data. Not available for single-participant estimation.'
        }
    
    def _calculate_composite_reliability(self,
                                       construct_name: str,
                                       feature_values: Dict[str, float],
                                       construct_def) -> Dict[str, float]:
        """
        Placeholder for composite reliability calculation.
        
        NOTE: This method is disabled because composite reliability requires
        indicator loadings from confirmatory factor analysis or multi-participant
        correlation data. Single-participant reliability estimation is not
        psychometrically valid.
        
        This method is retained for API compatibility but returns None.
        """
        return {
            'composite_reliability': None,
            'reliability_type': 'not_computable',
            'message': 'Composite reliability requires indicator loadings from CFA or multi-participant data. Not available for single-participant estimation.'
        }
    
    def _interpret_reliability(self, reliability: float) -> str:
        """Interpret reliability coefficient according to standard guidelines."""
        if reliability is None:
            return "Cannot calculate reliability"
        elif reliability >= 0.9:
            return "Excellent reliability"
        elif reliability >= 0.8:
            return "Good reliability"
        elif reliability >= 0.7:
            return "Acceptable reliability"
        elif reliability >= 0.6:
            return "Questionable reliability"
        elif reliability >= 0.5:
            return "Poor reliability"
        else:
            return "Unacceptable reliability"
    
    def _apply_circular_transform(self, hour_value: float) -> float:
        """
        Apply circular transformation to a single hour value.
        
        For single values, we convert to a standardized circular representation
        that preserves temporal relationships while remaining real-valued.
        
        Args:
            hour_value: Hour value (0-24)
            
        Returns:
            Real-valued circular representation suitable for aggregation
        """
        # Convert hour to angle on unit circle (0 to 2π)
        theta = (hour_value / 24.0) * 2 * math.pi
        
        # Map to real line using sine component
        # This preserves circular information while returning real values
        # Use sine component which gives values from -1 to 1
        # This represents the position on the circular scale
        return math.sin(theta)
    
    def _calculate_circular_sd(self, hour_values: List[float]) -> float:
        """
        Calculate true circular standard deviation for multiple hour values.
        
        Implements registry formula: sqrt(-2 * ln(R)) where R is mean resultant length
        
        Args:
            hour_values: List of hour values (0-24)
            
        Returns:
            Circular standard deviation
        """
        if not hour_values:
            return 0.0
        
        # Convert all hours to radians
        thetas = [(h / 24.0) * 2 * math.pi for h in hour_values]
        
        # Calculate sum of sin and cos components
        cos_sum = sum(math.cos(theta) for theta in thetas)
        sin_sum = sum(math.sin(theta) for theta in thetas)
        
        # Calculate mean resultant length R
        n = len(thetas)
        R = math.sqrt(cos_sum**2 + sin_sum**2) / n
        
        # Calculate circular standard deviation
        # Handle edge case where R is very close to 0
        if R < 1e-10:
            return math.pi  # Maximum possible circular SD
        
        circular_sd = math.sqrt(-2 * math.log(R))
        return circular_sd
    
    def _calculate_dispersion_interval(self,
                                      normalized_features: Dict[str, float],
                                      feature_weights: Dict[str, float]) -> Optional[Tuple[float, float]]:
        """
        Calculate weighted standard deviation of indicators as dispersion measure.
        
        NOTE: This is NOT a confidence interval. Indicators are not independent
        observations, so standard statistical CI formulas are invalid.
        This returns the weighted standard deviation as a simple dispersion measure.
        """
        if len(normalized_features) < 2:
            return None
        
        values = list(normalized_features.values())
        weights = [feature_weights.get(name, 1.0) for name in normalized_features]
        
        # Calculate weighted mean
        weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        
        # Calculate weighted standard deviation
        if len(values) > 1:
            weighted_variance = sum(w * (v - weighted_mean)**2 for v, w in zip(values, weights)) / sum(weights)
            weighted_sd = math.sqrt(weighted_variance)
            
            # Return mean ± weighted SD as simple dispersion interval
            return (weighted_mean - weighted_sd, weighted_mean + weighted_sd)
        
        return None
    
    def _generate_interpretation(self,
                                construct_name: str,
                                score: float,
                                quality_metrics: Dict[str, Any]) -> str:
        """
        Generate interpretation of construct score.
        
        Interpretation thresholds are normalization-method aware:
        - zscore: threshold at 0 (mean = 0)
        - minmax: threshold at 0.5 (mean ≈ 0.5) 
        - robust: threshold at 0 (median = 0)
        - none: no meaningful threshold
        """
        quality = quality_metrics['overall_quality']
        
        if quality < 0.5:
            return f"Low quality score for {construct_name} - interpret with caution"
        
        # Determine threshold based on normalization method
        if self.config.normalization_method == "zscore":
            threshold = 0.0
            high_desc = "elevated"
            low_desc = "reduced"
        elif self.config.normalization_method == "minmax":
            threshold = 0.5
            high_desc = "high"
            low_desc = "low"
        elif self.config.normalization_method == "robust":
            threshold = 0.0
            high_desc = "elevated"
            low_desc = "reduced"
        else:
            # No normalization or unknown method
            return f"Score for {construct_name}: {score:.2f} (no interpretation available)"
        
        # Interpret based on construct type and normalized score
        if construct_name == "behavioral_activation":
            if score > threshold:
                return f"{high_desc.capitalize()} behavioral activation - active and engaged"
            else:
                return f"{low_desc.capitalize()} behavioral activation - reduced activity levels"
        
        elif construct_name == "avoidance":
            if score > threshold:
                return f"{high_desc.capitalize()} avoidance - withdrawn or isolating behaviors"
            else:
                return f"{low_desc.capitalize()} avoidance - engaged and approach-oriented"
        
        elif construct_name == "social_engagement":
            if score > threshold:
                return f"{high_desc.capitalize()} social engagement - active social participation"
            else:
                return f"{low_desc.capitalize()} social engagement - reduced social interaction"
        
        elif construct_name == "routine_stability":
            if score > threshold:
                return f"{high_desc.capitalize()} routine stability - consistent daily patterns"
            else:
                return f"{low_desc.capitalize()} routine stability - irregular or chaotic patterns"
        
        else:
            return f"Score for {construct_name}: {score:.2f} ({high_desc} if > {threshold})"
    
    def get_construct_info(self, construct_name: str) -> Dict[str, Any]:
        """Get information about a construct."""
        construct_def = self.construct_registry.get_construct(construct_name)
        
        return {
            "name": construct_def.name,
            "description": construct_def.description,
            "measurement_model": construct_def.measurement_model,
            "aggregation_type": construct_def.aggregation_type,
            "features": [
                {
                    "name": f.name,
                    "weight": f.weight,
                    "aggregation": f.aggregation,
                    "validation_status": f.validation_status,
                    "description": f.description,
                    "direction": f.direction
                }
                for f in construct_def.features
            ]
        }
    
    def list_constructs(self) -> List[str]:
        """List all available constructs."""
        return self.construct_registry.list_constructs()
    
    def export_scores(self,
                     construct_scores: Dict[str, ConstructScore],
                     output_path: str,
                     format: str = "json") -> None:
        """Export construct scores to file."""
        if format.lower() == "json":
            self._export_json(construct_scores, output_path)
        elif format.lower() == "csv":
            self._export_csv(construct_scores, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, construct_scores: Dict[str, ConstructScore], output_path: str) -> None:
        """Export scores as JSON."""
        export_data = {}
        
        for construct_name, score in construct_scores.items():
            export_data[construct_name] = {
                'score': score.score,
                'normalized_score': score.normalized_score,
                'feature_scores': score.feature_scores,
                'quality_metrics': score.quality_metrics,
                'aggregation_parameters': score.aggregation_parameters,
                'timestamp': score.timestamp.isoformat(),
                'participant_id': score.participant_id,
                'dispersion_interval': score.dispersion_interval,
                'interpretation': score.interpretation
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_csv(self, construct_scores: Dict[str, ConstructScore], output_path: str) -> None:
        """Export scores as CSV."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'construct', 'score', 'normalized_score', 'overall_quality',
                'participant_id', 'timestamp', 'interpretation'
            ])
            
            # Data rows
            for construct_name, score in construct_scores.items():
                writer.writerow([
                    construct_name,
                    score.score,
                    score.normalized_score,
                    score.quality_metrics['overall_quality'],
                    score.participant_id,
                    score.timestamp.isoformat(),
                    score.interpretation
                ])
