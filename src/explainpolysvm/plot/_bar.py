import warnings
try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib is not installed.")
    pass
import numpy as np
from typing import List, Tuple

def bar(bar_heights: np.ndarray, labels: List[str], show: bool = True, figsize: Tuple[int] = (5, 4),
        xlabel: str = None, ylabel: str = None, title: str = None):
    """
    Create a bar chart.

    Parameters
    ----------
    bar_heights : Numpy array of floats
        Heights of bars.
    labels : List of strings
        List of labels to use for the x-ticks.
    show : Boolean
        Set to True (default) to run plt.show() at the very end of this function. Otherwise,
        return the figure for postprocessing.
    figsize : Tuple of two integers
        Size of the pyplot graph. Should be of the format [w, h] or (w, h) where w and h are integers. 
    xlabel : String
        (Optional) X-label to add to the plot
    ylabel : String
        (Optional) Y-label to add to the plot
    title : String
        (Optional) Title to add to the plot

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.bar(x=np.arange(len(bar_heights)), height=bar_heights, tick_label=labels)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlim([-1, len(bar_heights)])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
        
    if show:
        plt.show()
    else:
        return fig
