from .gbm_simulation import GBMSimulator
from .base_gbm import BaseGBM

__version__ = "0.1.0"
__all__ = ["GBMSimulator", "BaseGBM"]

# Import version information for the package
from .version import __version__  # Ensure version.py contains this variable

# Print the package version for confirmation (optional)
print(f"pygbm package version: {__version__}")