"""
Adaptive quality threshold system for dynamic quality assessment.

This module implements adaptive quality thresholds that adjust based on
data characteristics, construct type, and measurement model.

Product: Construct-Aligned Digital Phenotyping Toolkit
Purpose: Adaptive quality assessment for improved construct validity
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


class QualityRegime(Enum):
    """Quality assessment regimes based on data characteristics."""
    STRICT = "strict"      # High-quality data, low tolerance for issues
    MODERATE = "moderate"  # Typical data, balanced thresholds
    LENIENT = "lenient"    # Noisy data, higher tolerance for issues


@dataclass
class AdaptiveQualityConfig:
    """Configuration for adaptive quality assessment."""
    
    # Base quality thresholds
    base_min_quality: float = 0.5
    base_min_features: int = 2
    
    # Adaptive adjustment factors
    data_coverage_weight: float = 0.3
    construct_complexity_weight: float = 0.2
    measurement_model_weight: float = 0.2
    temporal_stability_weight: float = 0.3
    
    # Regime-specific adjustments
    strict_multiplier: float = 1.2
    moderate_multiplier: float = 1.0
    lenient_multiplier: float = 0.8
    
    # Minimum bounds
    absolute_min_quality: float = 0.3
    absolute_min_features: int = 1


class AdaptiveQualityAssessor:
    """
    Adaptive quality assessment system.
    
    This class implements dynamic quality thresholds that adjust based on
    data characteristics, construct complexity, and measurement model.
    """
    
    def __init__(self, config: Optional[AdaptiveQualityConfig] = None):
        """
        Initialize adaptive quality assessor.
        
        Args:
            config: Configuration for adaptive quality assessment
        """
        self.config = config or AdaptiveQualityConfig()
    
    def assess_data_regime(self, 
                           data_characteristics: Dict[str, Any]) -> QualityRegime:
        """
        Assess the quality regime based on data characteristics.
        
        Args:
            data_characteristics: Dictionary with data quality metrics
            
        Returns:
            QualityRegime classification
        """
        # Extract key characteristics
        coverage_ratio = data_characteristics.get('coverage_ratio', 0.5)
        missing_data_ratio = data_characteristics.get('missing_data_ratio', 0.0)
        data_quality_score = data_characteristics.get('data_quality_score', 0.5)
        temporal_consistency = data_characteristics.get('temporal_consistency', 0.5)
        
        # Calculate regime score
        regime_score = (
            coverage_ratio * 0.3 +
            (1 - missing_data_ratio) * 0.3 +
            data_quality_score * 0.2 +
            temporal_consistency * 0.2
        )
        
        # Classify regime
        if regime_score >= 0.8:
            return QualityRegime.STRICT
        elif regime_score >= 0.5:
            return QualityRegime.MODERATE
        else:
            return QualityRegime.LENIENT
    
    def calculate_adaptive_thresholds(self,
                                     construct_name: str,
                                     measurement_model: str,
                                     feature_count: int,
                                     data_regime: QualityRegime,
                                     construct_complexity: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate adaptive quality thresholds based on multiple factors.
        
        Args:
            construct_name: Name of the construct
            measurement_model: Measurement model (reflective/formative)
            feature_count: Number of features in the construct
            data_regime: Assessed quality regime
            construct_complexity: Optional complexity score (0-1)
            
        Returns:
            Dictionary with adaptive thresholds
        """
        # Base thresholds
        base_quality = self.config.base_min_quality
        base_features = self.config.base_min_features
        
        # Regime multiplier
        regime_multipliers = {
            QualityRegime.STRICT: self.config.strict_multiplier,
            QualityRegime.MODERATE: self.config.moderate_multiplier,
            QualityRegime.LENIENT: self.config.lenient_multiplier
        }
        regime_multiplier = regime_multipliers[data_regime]
        
        # Complexity adjustment
        if construct_complexity is None:
            # Estimate complexity from feature count and measurement model
            construct_complexity = self._estimate_construct_complexity(
                construct_name, measurement_model, feature_count
            )
        
        # Measurement model adjustment
        model_adjustment = self._get_model_adjustment(measurement_model)
        
        # Feature count adjustment
        feature_adjustment = self._get_feature_adjustment(feature_count)
        
        # Calculate adaptive quality threshold
        adaptive_quality = base_quality * regime_multiplier
        adaptive_quality *= (1 + construct_complexity * self.config.construct_complexity_weight)
        adaptive_quality *= model_adjustment
        adaptive_quality *= feature_adjustment
        
        # Apply bounds
        adaptive_quality = max(
            self.config.absolute_min_quality,
            min(1.0, adaptive_quality)
        )
        
        # Calculate adaptive feature threshold
        adaptive_features = max(
            self.config.absolute_min_features,
            int(base_features * regime_multiplier)
        )
        
        # Adjust for construct complexity
        if construct_complexity > 0.7:
            adaptive_features = max(adaptive_features, 3)
        
        return {
            'min_quality_threshold': adaptive_quality,
            'min_features_threshold': adaptive_features,
            'quality_regime': data_regime.value,
            'regime_multiplier': regime_multiplier,
            'construct_complexity': construct_complexity,
            'model_adjustment': model_adjustment,
            'feature_adjustment': feature_adjustment
        }
    
    def _estimate_construct_complexity(self,
                                     construct_name: str,
                                     measurement_model: str,
                                     feature_count: int) -> float:
        """
        Estimate construct complexity based on known characteristics.
        
        Args:
            construct_name: Name of the construct
            measurement_model: Measurement model type
            feature_count: Number of features
            
        Returns:
            Complexity score (0-1)
        """
        # Base complexity from feature count
        if feature_count <= 2:
            count_complexity = 0.2
        elif feature_count <= 4:
            count_complexity = 0.5
        else:
            count_complexity = 0.8
        
        # Measurement model complexity
        if measurement_model.lower() == "formative":
            model_complexity = 0.7  # Formative models are more complex
        elif measurement_model.lower() == "reflective":
            model_complexity = 0.3  # Reflective models are simpler
        else:
            model_complexity = 0.5  # Unknown/default
        
        # Construct-specific complexity
        construct_complexity_map = {
            "behavioral_activation": 0.4,
            "avoidance": 0.6,
            "social_engagement": 0.7,
            "routine_stability": 0.8
        }
        construct_complexity = construct_complexity_map.get(construct_name, 0.5)
        
        # Combine complexities
        overall_complexity = (
            count_complexity * 0.4 +
            model_complexity * 0.3 +
            construct_complexity * 0.3
        )
        
        return overall_complexity
    
    def _get_model_adjustment(self, measurement_model: str) -> float:
        """
        Get quality threshold adjustment based on measurement model.
        
        Args:
            measurement_model: Measurement model type
            
        Returns:
            Adjustment multiplier
        """
        if measurement_model.lower() == "reflective":
            return 1.1  # Higher quality standards for reflective models
        elif measurement_model.lower() == "formative":
            return 0.9  # Slightly lower standards for formative models
        else:
            return 1.0  # Default
    
    def _get_feature_adjustment(self, feature_count: int) -> float:
        """
        Get quality threshold adjustment based on feature count.
        
        Args:
            feature_count: Number of features
            
        Returns:
            Adjustment multiplier
        """
        if feature_count >= 5:
            return 1.1  # Higher standards for constructs with many features
        elif feature_count <= 2:
            return 0.9  # Lower standards for simple constructs
        else:
            return 1.0  # Default
    
    def evaluate_feature_quality(self,
                                feature_results: Dict[str, Any],
                                adaptive_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate feature quality against adaptive thresholds.
        
        Args:
            feature_results: Dictionary of feature extraction results
            adaptive_thresholds: Adaptive thresholds from calculate_adaptive_thresholds
            
        Returns:
            Quality evaluation results
        """
        min_quality = adaptive_thresholds['min_quality_threshold']
        min_features = adaptive_thresholds['min_features_threshold']
        
        # Extract quality scores
        feature_qualities = {}
        for feature_name, result in feature_results.items():
            quality = self._extract_feature_quality(result)
            feature_qualities[feature_name] = quality
        
        # Filter features by quality threshold
        high_quality_features = {
            name: quality for name, quality in feature_qualities.items()
            if quality >= min_quality
        }
        
        # Check minimum feature requirement
        meets_feature_requirement = len(high_quality_features) >= min_features
        
        # Calculate overall quality metrics
        if high_quality_features:
            mean_quality = np.mean(list(high_quality_features.values()))
            min_observed_quality = np.min(list(high_quality_features.values()))
        else:
            mean_quality = 0.0
            min_observed_quality = 0.0
        
        # Quality assessment
        quality_assessment = {
            'meets_quality_threshold': meets_feature_requirement and mean_quality >= min_quality,
            'meets_feature_requirement': meets_feature_requirement,
            'total_features': len(feature_results),
            'high_quality_features': len(high_quality_features),
            'feature_qualities': feature_qualities,
            'mean_quality': mean_quality,
            'min_quality': min_observed_quality,
            'quality_threshold': min_quality,
            'feature_threshold': min_features,
            'quality_regime': adaptive_thresholds['quality_regime']
        }
        
        return quality_assessment
    
    def _extract_feature_quality(self, feature_result: Dict[str, Any]) -> float:
        """
        Extract quality score from feature result.
        
        Args:
            feature_result: Feature extraction result
            
        Returns:
            Quality score (0-1)
        """
        # Try different quality keys
        quality_keys = ['quality', 'data_quality', 'quality_score', 'overall_quality']
        
        for key in quality_keys:
            if key in feature_result:
                quality = feature_result[key]
                if isinstance(quality, (int, float)) and 0 <= quality <= 1:
                    return float(quality)
        
        # Try nested quality metrics
        if 'quality_metrics' in feature_result:
            quality_metrics = feature_result['quality_metrics']
            if isinstance(quality_metrics, dict):
                for key in quality_keys:
                    if key in quality_metrics:
                        quality = quality_metrics[key]
                        if isinstance(quality, (int, float)) and 0 <= quality <= 1:
                            return float(quality)
        
        # Default quality if not found
        return 0.8  # Assume reasonable quality
    
    def generate_quality_report(self,
                               quality_assessment: Dict[str, Any],
                               adaptive_thresholds: Dict[str, Any]) -> str:
        """
        Generate a human-readable quality assessment report.
        
        Args:
            quality_assessment: Quality evaluation results
            adaptive_thresholds: Adaptive thresholds used
            
        Returns:
            Formatted quality report
        """
        regime = adaptive_thresholds['quality_regime']
        quality_threshold = adaptive_thresholds['min_quality_threshold']
        feature_threshold = adaptive_thresholds['min_features_threshold']
        
        meets_quality = quality_assessment['meets_quality_threshold']
        meets_features = quality_assessment['meets_feature_requirement']
        total_features = quality_assessment['total_features']
        high_quality_features = quality_assessment['high_quality_features']
        mean_quality = quality_assessment['mean_quality']
        
        # Build report
        report_lines = [
            f"QUALITY ASSESSMENT REPORT",
            f"=" * 40,
            f"Quality Regime: {regime.upper()}",
            f"Quality Threshold: {quality_threshold:.3f}",
            f"Feature Threshold: {feature_threshold}",
            f"",
            f"RESULTS:",
            f"Total Features: {total_features}",
            f"High-Quality Features: {high_quality_features}",
            f"Mean Quality: {mean_quality:.3f}",
            f"Meets Quality Standards: {'YES' if meets_quality else 'NO'}",
            f"Meets Feature Requirements: {'YES' if meets_features else 'NO'}",
            f""
        ]
        
        # Add interpretation
        if meets_quality:
            report_lines.append("ASSESSMENT: PASSED - Construct meets adaptive quality standards")
        else:
            if not meets_features:
                report_lines.append("ASSESSMENT: FAILED - Insufficient high-quality features")
            else:
                report_lines.append("ASSESSMENT: FAILED - Quality below adaptive threshold")
        
        # Add recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            f"- Data quality regime: {regime}",
            f"- Consider improving data coverage for higher quality assessment"
        ])
        
        if not meets_features:
            report_lines.append(f"- Need at least {feature_threshold} high-quality features")
        
        if mean_quality < quality_threshold:
            report_lines.append(f"- Improve feature quality to exceed {quality_threshold:.3f}")
        
        return "\n".join(report_lines)
