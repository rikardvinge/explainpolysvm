import pytest
import numpy as np
from .context import plot


import matplotlib

class TestPlot:
    def test_bar(self):
        """
        Verify that bar outputs a matplotlib figure with show=False.
        """
        bar_heights = np.array([1,2,3])
        labels = [f'Feature {i}' for i in [1,2,3]]
        xlabel = 'xlabel'
        ylabel = 'ylabel'
        title = 'title'
        show = False

        fig = plot.bar(bar_heights, labels,
                       xlabel=xlabel, ylabel=ylabel, 
                       title=title, show=show)
        assert isinstance(fig, matplotlib.figure.Figure)
    
    def test_waterfall(self):
        """
        Verify that waterfall outputs a matplotlib figure with show=False.
        """
        bar_widths = np.array([1,2,3])
        labels = [f'Feature {i}' for i in [1,2,3]]
        show = False

        # No bar labels, sum nor special xlims
        show_values = False
        show_sum = False
        xlim = None
        fig = plot.waterfall(bar_widths, labels, show=show,
                       show_values=show_values, show_sum=show_sum, xlim=xlim)
        assert isinstance(fig, matplotlib.figure.Figure)

        # No sum nor special xlims but bar labels
        show_values = True
        show_sum = False
        xlim = None
        fig = plot.waterfall(bar_widths, labels, show=show,
                       show_values=show_values, show_sum=show_sum, xlim=xlim)
        assert isinstance(fig, matplotlib.figure.Figure)

        # No special xlims but bar labels and sum 
        show_values = True
        show_sum = True
        xlim = None
        fig = plot.waterfall(bar_widths, labels, show=show,
                       show_values=show_values, show_sum=show_sum, xlim=xlim)
        assert isinstance(fig, matplotlib.figure.Figure)

        # All of bar labels, sum and special xlims
        show_values = True
        show_sum = True
        xlim = [-1,1]
        fig = plot.waterfall(bar_widths, labels, show=show,
                       show_values=show_values, show_sum=show_sum, xlim=xlim)
        assert isinstance(fig, matplotlib.figure.Figure)
