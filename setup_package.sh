#!/bin/bash

# Setup script to create all necessary __init__.py files
# Run this from project root: bash setup_package.sh

echo "Setting up Python package structure..."

# Create __init__.py in src/
cat > src/__init__.py << 'EOF'
"""
Solar Forecasting Package
"""
__version__ = "1.0.0"
EOF

# Create __init__.py in src/models/
cat > src/models/__init__.py << 'EOF'
"""
Forecasting models module
"""
from .base_model import BaseForecastModel, EnsembleModel

__all__ = ['BaseForecastModel', 'EnsembleModel']
EOF

# Create __init__.py in src/data/
cat > src/data/__init__.py << 'EOF'
"""
Data processing module
"""
from .preprocessing import SolarDataPreprocessor
from .feature_engineering import SolarFeatureEngineer
from .weather_regimes import WeatherRegimeDetector

__all__ = ['SolarDataPreprocessor', 'SolarFeatureEngineer', 'WeatherRegimeDetector']
EOF

# Create __init__.py in src/utils/
cat > src/utils/__init__.py << 'EOF'
"""
Utility functions module
"""
from .logger import setup_logger, ExperimentLogger
from .config import Config

__all__ = ['setup_logger', 'ExperimentLogger', 'Config']
EOF

# Create __init__.py in src/ensemble/
cat > src/ensemble/__init__.py << 'EOF'
"""
Ensemble methods module
"""
__all__ = []
EOF

# Create __init__.py in src/uncertainty/
cat > src/uncertainty/__init__.py << 'EOF'
"""
Uncertainty quantification module
"""
__all__ = []
EOF

# Create __init__.py in src/explainability/
cat > src/explainability/__init__.py << 'EOF'
"""
Explainability module
"""
__all__ = []
EOF

# Create __init__.py in src/evaluation/
cat > src/evaluation/__init__.py << 'EOF'
"""
Evaluation metrics module
"""
__all__ = []
EOF

echo "Package structure created successfully!"
echo ""
echo "Created __init__.py files in:"
echo "  - src/"
echo "  - src/models/"
echo "  - src/data/"
echo "  - src/utils/"
echo "  - src/ensemble/"
echo "  - src/uncertainty/"
echo "  - src/explainability/"
echo "  - src/evaluation/"
echo ""
echo "Now you can run: export PYTHONPATH=/data/Sajjan_Singh/DSP_Project:$PYTHONPATH"