"""
Setup script for Psyconstruct - Digital Phenotyping for Mental Health.

This script handles package installation, dependencies, and distribution.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    """Read README file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line:
                        requirements.append(line)
    
    return requirements

# Read development requirements
def read_dev_requirements():
    """Read development requirements from requirements-dev.txt."""
    dev_requirements_path = os.path.join(os.path.dirname(__file__), 'requirements-dev.txt')
    dev_requirements = []
    
    if os.path.exists(dev_requirements_path):
        with open(dev_requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip the -r requirements.txt line and comments
                if line and not line.startswith('#') and not line.startswith('-r'):
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line:
                        dev_requirements.append(line)
    
    return dev_requirements

# Package metadata
NAME = "psyconstruct"
VERSION = "1.0.0"
DESCRIPTION = "Digital phenotyping framework for mental health research and clinical practice"
LONG_DESCRIPTION = read_readme()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

# Author information
AUTHOR = "Psyconstruct Development Team"
AUTHOR_EMAIL = "psyconstruct@example.com"
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL

# Project URLs
PROJECT_URLS = {
    "Documentation": "https://psyconstruct.readthedocs.io/",
    "Source": "https://github.com/your-repo/psyconstruct",
    "Tracker": "https://github.com/your-repo/psyconstruct/issues",
    "Changelog": "https://github.com/your-repo/psyconstruct/blob/main/CHANGELOG.md",
}

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Package classification
CLASSIFIERS = [
    # Development Status
    "Development Status :: 5 - Production/Stable",
    
    # Intended Audience
    "Intended Audience :: Developers",
    "Intended Audience :: Healthcare Industry",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
    
    # Topic
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Health Care",
    "Topic :: Software Development :: Libraries",
    
    # License
    "License :: OSI Approved :: MIT License",
    
    # Programming Language
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    
    # Operating System
    "Operating System :: OS Independent",
    
    # Environment
    "Environment :: Console",
    "Environment :: Other Environment",
    
    # Natural Language
    "Natural Language :: English",
    
    # Typing
    "Typing :: Typed",
]

# Keywords for package discovery
KEYWORDS = [
    "digital phenotyping",
    "mental health",
    "behavioral analysis",
    "psychological constructs",
    "mobile sensing",
    "smartphone data",
    "clinical research",
    "machine learning",
    "data science",
    "healthcare",
    "depression",
    "anxiety",
    "behavioral activation",
    "avoidance",
    "social engagement",
    "routine stability"
]

# Package requirements
INSTALL_REQUIRES = read_requirements()
EXTRAS_REQUIRE = {
    "dev": read_dev_requirements(),
    "test": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "pytest-mock>=3.8.0",
        "pytest-xdist>=2.5.0",
    ],
    "docs": [
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
        "sphinx-autodoc-typehints>=1.17.0",
        "myst-parser>=0.18.0",
        "sphinxcontrib-napoleon>=0.7",
    ],
    "geospatial": [
        "shapely>=1.8.0",
        "geopandas>=0.11.0",
    ],
    "ml": [
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        "mlflow>=1.27.0",
        "optuna>=3.0.0",
    ],
    "database": [
        "sqlalchemy>=1.4.0",
        "alembic>=1.8.0",
        "psycopg2-binary>=2.9.0",
        "pymongo>=4.2.0",
    ],
    "cloud": [
        "boto3>=1.24.0",
        "google-cloud-storage>=2.5.0",
    ],
    "api": [
        "fastapi>=0.78.0",
        "uvicorn>=0.18.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "ipykernel>=6.15.0",
        "ipywidgets>=7.7.0",
        "notebook>=6.4.0",
        "jupyterlab>=3.4.0",
    ],
    "visualization": [
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        "rich>=12.5.0",
    ],
    "performance": [
        "memory-profiler>=0.60.0",
        "line-profiler>=3.5.0",
        "py-spy>=0.3.12",
    ],
    "security": [
        "bandit>=1.7.0",
        "safety>=2.1.0",
    ]
}

# Add 'all' extra that includes all optional dependencies
EXTRAS_REQUIRE["all"] = [
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
]

# Entry points for command line tools
ENTRY_POINTS = {
    "console_scripts": [
        "psyconstruct=psyconstruct.cli:main",
        "psyconstruct-validate=psyconstruct.validation:main",
        "psyconstruct-analyze=psyconstruct.analysis:main",
    ],
}

# Package discovery
PACKAGES = find_packages(exclude=["tests", "tests.*", "docs", "docs.*"])

# Include additional files
PACKAGE_DATA = {
    "psyconstruct": [
        "data/*.json",
        "data/*.yaml",
        "data/*.yml",
        "constructs/registry.json",
        "constructs/registry.yaml",
    ],
}

# Data files to include
DATA_FILES = [
    ("share/psyconstruct/examples", [
        "examples/behavioral_activation_example.py",
        "examples/avoidance_features_example.py",
        "examples/social_engagement_example.py",
        "examples/routine_stability_example.py",
        "examples/construct_aggregation_example.py",
    ]),
    ("share/psyconstruct/docs", [
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
    ]),
]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    
    url=PROJECT_URLS["Source"],
    project_urls=PROJECT_URLS,
    
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    data_files=DATA_FILES,
    
    include_package_data=True,
    zip_safe=False,
    
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    entry_points=ENTRY_POINTS,
    
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    
    # Test suite configuration
    test_suite="tests",
    tests_require=EXTRAS_REQUIRE["test"],
    
    # Additional metadata
    platforms=["any"],
    license="MIT",
)

# Installation instructions:
"""
# Standard installation
pip install psyconstruct

# Installation with development tools
pip install psyconstruct[dev]

# Installation with all optional dependencies
pip install psyconstruct[all]

# Installation from source
git clone https://github.com/your-repo/psyconstruct.git
cd psyconstruct
pip install -e .

# Development installation
git clone https://github.com/your-repo/psyconstruct.git
cd psyconstruct
pip install -e .[dev]

# Installation with specific extras
pip install psyconstruct[geospatial,ml,jupyter]

# Installation for testing
pip install psyconstruct[test]
"""
