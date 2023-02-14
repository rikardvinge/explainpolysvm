from .expsvm import ExPSVM, InteractionUtils, dict2array
import warnings

try:
    import matplotlib
    
    # Import plots
    from .plot._waterfall import waterfall

except ImportError:
    warnings.warn("matplotlib is not installed and plotting is disabled. Please install using `pip install matplotlib`.")