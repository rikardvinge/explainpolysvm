import warnings
try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib is not installed.")
    pass
import numpy as np
from typing import List, Tuple

def bar(bar_heights: np.ndarray, labels: List[str], show: bool = True, figsize: Tuple[int] = (5, 4),
        xlabel: str = None, ylabel: str = None, title: str = None,
        yerr: np.ndarray = None, markers: np.ndarray = None, marker_label: str = None,
        rotation: int = 90):
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
    yerr : Numpy array of floats
        (Optional) Spread of each bar, drawn as error bars.
    markers : Numpy array of floats
        (Optional) One value per bar, drawn as a marker on top of the bars. Used to show a second
        quantity, such as the signed mean beside bars of the mean magnitude.
    marker_label : String
        (Optional) Label of the markers. A legend is added to the plot if provided.
    rotation : Int
        Rotation of the x-tick labels in degrees. Default is 90, which keeps long interaction names
        readable. Use 0 for short labels, such as the names of the polynomial degrees.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(len(bar_heights))
    ax.bar(x=x, height=bar_heights, tick_label=labels, yerr=yerr,
           capsize=3 if yerr is not None else 0)

    # Add markers of a second quantity, one per bar.
    if markers is not None:
        ax.plot(x, markers, linestyle='none', marker='D', color='k', label=marker_label)
        if marker_label is not None:
            ax.legend()

    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
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
