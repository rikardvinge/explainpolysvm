import warnings
try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib is not installed.")
    pass
import numpy as np
from typing import List, Tuple


def box(box_data: np.ndarray, labels: List[str], show: bool = True, figsize: Tuple[int] = (5, 4),
        xlabel: str = None, ylabel: str = None, title: str = None, show_zero: bool = True,
        rotation: int = 90, hline: float = None, hline_label: str = None,
        hline_color: str = 'tab:orange'):
    """
    Create a box plot, one box per column of box_data.

    Parameters
    ----------
    box_data : Numpy ndarray of shape (n_observations, n_boxes)
        Values to summarize. One box is drawn per column.
    labels : List of strings
        List of labels to use for the x-ticks. One label per column of box_data.
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
    show_zero : Boolean
        Set to True (default) to add a horizontal line at zero, separating positive from negative
        contributions.
    rotation : Int
        Rotation of the x-tick labels in degrees. Default is 90. Use 0 for short labels, such as the
        names of the polynomial degrees.
    hline : Float
        (Optional) Value of a horizontal line drawn across the plot. Used to show a constant beside
        the boxes, such as the intercept of the model.
    hline_label : String
        (Optional) Label of the horizontal line. A legend is added to the plot if provided.
    hline_color : String
        Matplotlib color of the horizontal line. Default is 'tab:orange', the color used for the
        medians of the boxes.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    # Create box plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    n_boxes = box_data.shape[1]
    ax.boxplot([box_data[:, ind] for ind in np.arange(n_boxes)])

    # Set the labels separately since the name of the boxplot label argument has changed
    # between matplotlib versions.
    ax.set_xticks(ticks=np.arange(1, n_boxes + 1), labels=labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    if show_zero:
        ax.axhline(0, linestyle='--', color='tab:gray', zorder=-np.inf)

    # Add a constant as a line across the plot, drawn like the medians of the boxes.
    if hline is not None:
        ax.axhline(hline, color=hline_color, label=hline_label)
        if hline_label is not None:
            ax.legend()

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
