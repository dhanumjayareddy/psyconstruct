"""
Provenance tracking utility for psyconstruct package.

This module provides comprehensive provenance tracking for all operations
performed during digital phenotyping feature extraction, ensuring scientific
reproducibility and audit trail capabilities.

Product: Construct-Aligned Digital Phenotyping Toolkit
Purpose: Scientific reproducibility through provenance tracking
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid


@dataclass
class OperationProvenance:
    """Provenance record for a single operation."""
    
    operation_id: str
    operation_type: str
    timestamp: str
    duration_seconds: float
    input_parameters: Dict[str, Any]
    output_summary: Dict[str, Any]
    software_version: str
    random_seed: Optional[int] = None
    parent_operations: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.parent_operations is None:
            self.parent_operations = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FeatureExtractionProvenance:
    """Provenance record for feature extraction operations."""
    
    feature_name: str
    construct: str
    extraction_timestamp: str
    input_data_summary: Dict[str, Any]
    computation_parameters: Dict[str, Any]
    result_summary: Dict[str, Any]
    data_quality_metrics: Dict[str, Any]
    algorithm_version: str
    computational_hash: str
    
    def __post_init__(self):
        """Validate required fields."""
        if not self.feature_name:
            raise ValueError("Feature name is required")
        if not self.construct:
            raise ValueError("Construct name is required")


@dataclass
class ConstructAggregationProvenance:
    """Provenance record for construct aggregation operations."""
    
    construct_name: str
    aggregation_timestamp: str
    input_features: List[str]
    aggregation_method: str
    feature_weights: Dict[str, float]
    normalization_applied: bool
    measurement_model: str
    result_summary: Dict[str, Any]
    aggregation_hash: str
    
    def __post_init__(self):
        """Validate weight sums."""
        if self.feature_weights:
            total_weight = sum(self.feature_weights.values())
            if not (0.99 <= total_weight <= 1.01):
                raise ValueError(f"Feature weights must sum to 1.0, got {total_weight:.3f}")


class ProvenanceTracker:
    """
    Comprehensive provenance tracking for digital phenotyping operations.
    
    This class provides methods to track, store, and retrieve provenance information
    for all operations performed during feature extraction and construct aggregation.
    It ensures reproducibility by maintaining detailed audit trails and computational
    hashes for verification.
    
    Attributes:
        session_id: Unique identifier for the current tracking session
        operations: List of all operation provenance records
        feature_extractions: List of feature extraction provenance records
        construct_aggregations: List of construct aggregation provenance records
        session_metadata: Additional metadata about the tracking session
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize provenance tracker.
        
        Args:
            session_id: Unique session identifier. If None, generates UUID.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.operations: List[OperationProvenance] = []
        self.feature_extractions: List[FeatureExtractionProvenance] = []
        self.construct_aggregations: List[ConstructAggregationProvenance] = []
        self.session_metadata = {
            'session_start': datetime.now().isoformat(),
            'software_version': '0.1.0',  # Should match package version
            'python_version': self._get_python_version(),
            'platform': self._get_platform_info()
        }
    
    def start_operation(self, 
                       operation_type: str,
                       input_parameters: Dict[str, Any],
                       parent_operations: Optional[List[str]] = None) -> str:
        """
        Start tracking a new operation.
        
        Feature Name: Operation Tracking
        Construct: Provenance (utility)
        Mathematical Definition: Generate unique operation identifier and record start time
        Formal Equation: operation_id = UUID5(session_id, operation_type, timestamp)
        Assumptions: Operation types are standardized and identifiable
        Limitations: Cannot track operations that don't use this tracking system
        Edge Cases: Concurrent operations, nested operations, operation failures
        Output Schema: Operation identifier for later completion tracking
        
        Args:
            operation_type: Type of operation (e.g., 'harmonize_gps_data')
            input_parameters: Parameters passed to the operation
            parent_operations: List of parent operation IDs for dependency tracking
            
        Returns:
            Operation identifier for tracking completion
        """
        operation_id = str(uuid.uuid4())  # Use simple UUID4 instead of UUID5
        
        # Record operation start
        operation = OperationProvenance(
            operation_id=operation_id,
            operation_type=operation_type,
            timestamp=datetime.now().isoformat(),
            duration_seconds=0.0,  # Will be updated on completion
            input_parameters=input_parameters,
            output_summary={},  # Will be updated on completion
            software_version=self.session_metadata['software_version'],
            parent_operations=parent_operations or []
        )
        
        self.operations.append(operation)
        return operation_id
    
    def complete_operation(self,
                          operation_id: str,
                          output_summary: Dict[str, Any],
                          duration_seconds: float,
                          random_seed: Optional[int] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Complete tracking of an operation.
        
        Args:
            operation_id: Identifier returned by start_operation
            output_summary: Summary of operation outputs
            duration_seconds: Total operation duration
            random_seed: Random seed used (if applicable)
            metadata: Additional metadata about the operation
            
        Raises:
            ValueError: If operation_id is not found
        """
        # Find the operation
        for operation in self.operations:
            if operation.operation_id == operation_id:
                operation.output_summary = output_summary
                operation.duration_seconds = duration_seconds
                operation.random_seed = random_seed
                operation.metadata = metadata or {}
                return
        
        raise ValueError(f"Operation ID not found: {operation_id}")
    
    def record_feature_extraction(self,
                                 feature_name: str,
                                 construct: str,
                                 input_data_summary: Dict[str, Any],
                                 computation_parameters: Dict[str, Any],
                                 result_summary: Dict[str, Any],
                                 data_quality_metrics: Dict[str, Any],
                                 algorithm_version: str = '1.0.0') -> str:
        """
        Record provenance for feature extraction.
        
        Feature Name: Feature Extraction Provenance
        Construct: Provenance (utility)
        Mathematical Definition: Generate computational hash and record extraction details
        Formal Equation: hash = SHA256(feature_name + construct + parameters + data_hash)
        Assumptions: Feature extraction follows standardized parameter format
        Limitations: Cannot detect modifications to data after extraction
        Edge Cases: Missing data quality metrics, parameter variations, algorithm changes
        Output Schema: Feature extraction provenance record with computational hash
        
        Args:
            feature_name: Name of the extracted feature
            construct: Psychological construct the feature belongs to
            input_data_summary: Summary of input data characteristics
            computation_parameters: Parameters used for feature computation
            result_summary: Summary of extraction results
            data_quality_metrics: Quality metrics for the extraction
            algorithm_version: Version of the extraction algorithm
            
        Returns:
            Computational hash for verification
        """
        # Generate computational hash
        hash_input = {
            'feature_name': feature_name,
            'construct': construct,
            'parameters': computation_parameters,
            'data_summary': input_data_summary,
            'algorithm_version': algorithm_version
        }
        computational_hash = self._generate_hash(hash_input)
        
        # Create provenance record
        provenance = FeatureExtractionProvenance(
            feature_name=feature_name,
            construct=construct,
            extraction_timestamp=datetime.now().isoformat(),
            input_data_summary=input_data_summary,
            computation_parameters=computation_parameters,
            result_summary=result_summary,
            data_quality_metrics=data_quality_metrics,
            algorithm_version=algorithm_version,
            computational_hash=computational_hash
        )
        
        self.feature_extractions.append(provenance)
        return computational_hash
    
    def record_construct_aggregation(self,
                                   construct_name: str,
                                   input_features: List[str],
                                   aggregation_method: str,
                                   feature_weights: Dict[str, float],
                                   normalization_applied: bool,
                                   measurement_model: str,
                                   result_summary: Dict[str, Any]) -> str:
        """
        Record provenance for construct aggregation.
        
        Feature Name: Construct Aggregation Provenance
        Construct: Provenance (utility)
        Mathematical Definition: Generate aggregation hash and record aggregation details
        Formal Equation: hash = SHA256(construct_name + features + weights + method + model)
        Assumptions: Aggregation follows defined measurement model specifications
        Limitations: Cannot verify correctness of aggregation logic
        Edge Cases: Weight variations, normalization differences, model changes
        Output Schema: Construct aggregation provenance record with verification hash
        
        Args:
            construct_name: Name of the aggregated construct
            input_features: List of input feature names
            aggregation_method: Method used for aggregation
            feature_weights: Weights applied to each feature
            normalization_applied: Whether normalization was applied
            measurement_model: Measurement model (reflective/formative)
            result_summary: Summary of aggregation results
            
        Returns:
            Aggregation hash for verification
        """
        # Generate aggregation hash
        hash_input = {
            'construct_name': construct_name,
            'input_features': sorted(input_features),
            'aggregation_method': aggregation_method,
            'feature_weights': feature_weights,
            'normalization_applied': normalization_applied,
            'measurement_model': measurement_model
        }
        aggregation_hash = self._generate_hash(hash_input)
        
        # Create provenance record
        provenance = ConstructAggregationProvenance(
            construct_name=construct_name,
            aggregation_timestamp=datetime.now().isoformat(),
            input_features=input_features,
            aggregation_method=aggregation_method,
            feature_weights=feature_weights,
            normalization_applied=normalization_applied,
            measurement_model=measurement_model,
            result_summary=result_summary,
            aggregation_hash=aggregation_hash
        )
        
        self.construct_aggregations.append(provenance)
        return aggregation_hash
    
    def get_operation_provenance(self, operation_id: str) -> Optional[OperationProvenance]:
        """
        Get provenance for a specific operation.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            Operation provenance record or None if not found
        """
        for operation in self.operations:
            if operation.operation_id == operation_id:
                return operation
        return None
    
    def get_feature_provenance(self, feature_name: str) -> List[FeatureExtractionProvenance]:
        """
        Get provenance records for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            List of feature extraction provenance records
        """
        return [f for f in self.feature_extractions if f.feature_name == feature_name]
    
    def get_construct_provenance(self, construct_name: str) -> List[ConstructAggregationProvenance]:
        """
        Get provenance records for a specific construct.
        
        Args:
            construct_name: Name of the construct
            
        Returns:
            List of construct aggregation provenance records
        """
        return [c for c in self.construct_aggregations if c.construct_name == construct_name]
    
    def verify_reproducibility(self, 
                             feature_name: str,
                             construct: str,
                             computation_parameters: Dict[str, Any],
                             expected_hash: str) -> bool:
        """
        Verify that feature extraction is reproducible.
        
        Args:
            feature_name: Name of the feature
            construct: Psychological construct
            computation_parameters: Parameters used for computation
            expected_hash: Expected computational hash
            
        Returns:
            True if reproducible, False otherwise
        """
        # Generate hash for current parameters
        hash_input = {
            'feature_name': feature_name,
            'construct': construct,
            'parameters': computation_parameters,
            'algorithm_version': '1.0.0'  # Should match the original
        }
        current_hash = self._generate_hash(hash_input)
        
        return current_hash == expected_hash
    
    def export_provenance(self, 
                         output_path: Optional[Path] = None,
                         format_type: str = 'json') -> Dict[str, Any]:
        """
        Export complete provenance record.
        
        Args:
            output_path: Path to save provenance file. If None, returns as dict.
            format_type: Export format ('json' or 'yaml')
            
        Returns:
            Complete provenance record as dictionary
        """
        # Compile complete provenance
        complete_provenance = {
            'session_metadata': self.session_metadata,
            'session_id': self.session_id,
            'session_end': datetime.now().isoformat(),
            'operations': [asdict(op) for op in self.operations],
            'feature_extractions': [asdict(fe) for fe in self.feature_extractions],
            'construct_aggregations': [asdict(ca) for ca in self.construct_aggregations],
            'summary_statistics': self._generate_summary_statistics()
        }
        
        # Save to file if path provided
        if output_path:
            if format_type == 'json':
                with open(output_path, 'w') as f:
                    json.dump(complete_provenance, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        
        return complete_provenance
    
    def import_provenance(self, provenance_file: Path) -> None:
        """
        Import provenance from file.
        
        Args:
            provenance_file: Path to provenance file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not provenance_file.exists():
            raise FileNotFoundError(f"Provenance file not found: {provenance_file}")
        
        try:
            with open(provenance_file, 'r') as f:
                imported_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid provenance file format: {e}")
        
        # Import session metadata
        self.session_metadata.update(imported_data.get('session_metadata', {}))
        self.session_id = imported_data.get('session_id', self.session_id)
        
        # Import operations
        for op_data in imported_data.get('operations', []):
            operation = OperationProvenance(**op_data)
            self.operations.append(operation)
        
        # Import feature extractions
        for fe_data in imported_data.get('feature_extractions', []):
            feature_extraction = FeatureExtractionProvenance(**fe_data)
            self.feature_extractions.append(feature_extraction)
        
        # Import construct aggregations
        for ca_data in imported_data.get('construct_aggregations', []):
            construct_aggregation = ConstructAggregationProvenance(**ca_data)
            self.construct_aggregations.append(construct_aggregation)
    
    def clear_provenance(self) -> None:
        """Clear all provenance records."""
        self.operations.clear()
        self.feature_extractions.clear()
        self.construct_aggregations.clear()
        self.session_metadata['session_start'] = datetime.now().isoformat()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of the current tracking session.
        
        Returns:
            Session summary with key statistics
        """
        return {
            'session_id': self.session_id,
            'session_duration_seconds': (
                datetime.now() - datetime.fromisoformat(self.session_metadata['session_start'])
            ).total_seconds(),
            'total_operations': len(self.operations),
            'total_feature_extractions': len(self.feature_extractions),
            'total_construct_aggregations': len(self.construct_aggregations),
            'unique_features_extracted': len(set(f.feature_name for f in self.feature_extractions)),
            'unique_constructs_processed': len(set(c.construct_name for c in self.construct_aggregations))
        }
    
    # Private helper methods
    
    def _generate_hash(self, data: Dict[str, Any]) -> str:
        """Generate SHA256 hash for given data."""
        # Convert data to JSON string for consistent hashing
        data_string = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _get_python_version(self) -> str:
        """Get Python version information."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_platform_info(self) -> str:
        """Get platform information."""
        import platform
        return f"{platform.system()} {platform.release()}"
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the provenance record."""
        # Operation statistics
        operation_types = {}
        total_duration = 0.0
        for op in self.operations:
            op_type = op.operation_type
            operation_types[op_type] = operation_types.get(op_type, 0) + 1
            total_duration += op.duration_seconds
        
        # Feature statistics
        feature_counts = {}
        construct_counts = {}
        for fe in self.feature_extractions:
            feature_counts[fe.feature_name] = feature_counts.get(fe.feature_name, 0) + 1
            construct_counts[fe.construct] = construct_counts.get(fe.construct, 0) + 1
        
        return {
            'operation_statistics': {
                'total_operations': len(self.operations),
                'total_duration_seconds': total_duration,
                'operation_type_counts': operation_types,
                'average_duration_seconds': total_duration / len(self.operations) if self.operations else 0.0
            },
            'feature_statistics': {
                'total_extractions': len(self.feature_extractions),
                'unique_features': len(feature_counts),
                'feature_extraction_counts': feature_counts,
                'construct_extraction_counts': construct_counts
            },
            'construct_statistics': {
                'total_aggregations': len(self.construct_aggregations),
                'unique_constructs': len(set(c.construct_name for c in self.construct_aggregations)),
                'measurement_models': list(set(c.measurement_model for c in self.construct_aggregations))
            }
        }


# Global provenance tracker instance
_global_tracker: Optional[ProvenanceTracker] = None


def get_provenance_tracker(session_id: Optional[str] = None) -> ProvenanceTracker:
    """
    Get global provenance tracker instance.
    
    Args:
        session_id: Optional session identifier for new tracker
        
    Returns:
        Provenance tracker instance
    """
    global _global_tracker
    
    if _global_tracker is None or session_id is not None:
        _global_tracker = ProvenanceTracker(session_id)
    
    return _global_tracker


def track_operation(operation_type: str, input_parameters: Dict[str, Any]):
    """
    Decorator for automatic operation tracking.
    
    Args:
        operation_type: Type of operation being tracked
        input_parameters: Parameters to track
        
    Returns:
        Decorated function with provenance tracking
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = get_provenance_tracker()
            
            # Start operation tracking
            operation_id = tracker.start_operation(
                operation_type=operation_type,
                input_parameters=input_parameters
            )
            
            try:
                # Execute function
                start_time = datetime.now()
                result = func(*args, **kwargs)
                end_time = datetime.now()
                
                # Complete operation tracking
                duration = (end_time - start_time).total_seconds()
                output_summary = {
                    'success': True,
                    'output_type': type(result).__name__,
                    'execution_time': duration
                }
                
                tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary=output_summary,
                    duration_seconds=duration
                )
                
                return result
                
            except Exception as e:
                # Record failed operation
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                output_summary = {
                    'success': False,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'execution_time': duration
                }
                
                tracker.complete_operation(
                    operation_id=operation_id,
                    output_summary=output_summary,
                    duration_seconds=duration
                )
                
                raise
        
        return wrapper
    return decorator
