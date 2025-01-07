import warnings
try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib is not installed.")
    pass
import numpy as np
from typing import List, Tuple


def waterfall(bar_widths: np.ndarray, labels: List[str], show: bool = True,
              show_values: bool = False, show_sum: bool = False,
              figsize: Tuple[int] = (5, 4), xlim: List[float] = None,
              positive_color: 'str' = 'tab:red', negative_color: 'str' = 'tab:blue'):
    """
    Create a waterfall chart.

    Parameters
    ----------
    bar_widths : np.ndarray
        Widths of bins. Should have shape (n_original_features,).
    labels : List of strings
        List of labels for the bars
    show : Boolean
        Set to True (default) to run plt.show() at the very end of this function. Otherwise,
        return the figure for postprocessing.
    show_values : Boolean
        Set to True to add label with the bar width beside each bar. Default is False.
    show_sum : Boolean
        Set to True to add the sum of the bars at the bottom of the graph. Default is False
    figsize : Tuple of two integers
        Size of the pyplot graph. Should be of the format [w, h] or (w, h) where w and h are integers.
    xlim : List of two integers or None
        Custom limits to x-axis. This is useful to prettify the plots in case show_labels=True. This is due to
        the difficulties to get the extents of pyplot.text text boxes in a useful format. This input may be
        removed in the future in case a way of ensuring that the labels are confined to the inside of the
        drawing area is found.
    positive_color : str
        Matplotlib color to use for the interactions with positive contribution. Default is 'tab:blue'.
    negative_color : str
        Matplotlib color to use for the interactions with negative contribution. Default is 'tab:red'.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    # Instantiate plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    n_bars = len(bar_widths)

    ymin = -n_bars
    if show_sum:
        ymin -= 1
    ymax = 1
    yticks = np.flip(np.arange(-n_bars, 1))

    current_width = 0
    for w, y, l in zip(bar_widths, yticks, labels):
        # Plot bar
        c = positive_color if w > 0 else negative_color
        bar = ax.barh(y, w, left=current_width, color=c, align='center')

        # Add value label
        if show_values:
            add_sign = '+' if w > 0 else ''
            ax.bar_label(bar, [add_sign + '{:.1e}'.format(w)], label_type='edge', color=c)

        if (n_bars > 1) & (y < 0):
            # Plot vertical dashed line to previous bar
            ax.vlines(x=current_width, ymin=y - 0.4, ymax=y + 1.4, linestyle='-', color='black')

        current_width += w

    # Add sum of all bars
    if show_sum:
        bar_sum = np.sum(bar_widths)
        ax.vlines(bar_sum, ymin+1, 1.4 - n_bars, color='k', linestyle='--', clip_on=False)

        # Write decision function value
        ax.text(bar_sum, ymin+1, 'f(x)={:.2e}'.format(bar_sum), ha='center', va='top')

    # Add vertical line at x=0
    ax.vlines(0, ymin, 1, linestyle='--', color='tab:gray', zorder=-np.inf)

    # Prettify graph
    ax.set_yticks(ticks=-np.arange(n_bars), labels=labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Decision function value')
    ax.set_ylim([ymin, ymax])
    if xlim is not None:
        ax.set_xlim(xlim)

    if show:
        plt.show()
    else:
        return fig
