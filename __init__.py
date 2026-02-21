"""
Psyconstruct: Construct-Aligned Digital Phenotyping Toolkit

A Python package for theory-grounded smartphone-derived behavioral feature
extraction aligned with psychological constructs.

This package implements 14 theory-grounded features across four constructs:
- Behavioral Activation (BA)
- Avoidance (AV) 
- Social Engagement (SE)
- Routine Stability (RS)

Purpose: Research reproducibility and transparent feature extraction
"""

__version__ = "0.1.0"
__author__ = "Psyconstruct Development Team"
__email__ = "contact@psyconstruct.org"

# Import main configuration
from .config import PsyconstructConfig, get_config, DEFAULT_CONFIG

# Import core classes (will be added as modules are implemented)
# from .features import *
# from .constructs import *
# from .validation import *

__all__ = [
    "PsyconstructConfig",
    "get_config", 
    "DEFAULT_CONFIG"
]
