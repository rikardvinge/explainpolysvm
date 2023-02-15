try:
    import matplotlib
except ImportError:
    raise ImportError("matplotlib is not installed. Please install using `pip install matplotlib`.")

from ._waterfall import waterfall
from ._bar import bar