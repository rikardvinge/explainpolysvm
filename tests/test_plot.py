import pytest
import numpy as np
from explainpolysvm import plot


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
    
    def test_bar_errorbars_and_markers(self):
        """
        Verify that bar outputs a matplotlib figure when error bars and markers are added.
        """
        bar_heights = np.array([1, 2, 3])
        labels = [f'Feature {i}' for i in [1, 2, 3]]
        yerr = np.array([0.1, 0.2, 0.3])
        markers = np.array([0.5, -1.5, 2.5])
        show = False

        # Error bars only
        fig = plot.bar(bar_heights, labels, yerr=yerr, show=show)
        assert isinstance(fig, matplotlib.figure.Figure)

        # Markers only, without a legend
        fig = plot.bar(bar_heights, labels, markers=markers, show=show)
        assert isinstance(fig, matplotlib.figure.Figure)

        # Both, with a legend for the markers
        fig = plot.bar(bar_heights, labels, yerr=yerr, markers=markers,
                       marker_label='Signed mean', show=show)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_bar_rotation(self):
        """
        Verify that bar rotates the x-tick labels as requested, and rotates them 90 degrees by default.
        """
        bar_heights = np.array([1, 2, 3])
        labels = [f'Feature {i}' for i in [1, 2, 3]]

        fig = plot.bar(bar_heights, labels, show=False)
        assert np.all([label.get_rotation() == 90 for label in fig.axes[0].get_xticklabels()])

        fig = plot.bar(bar_heights, labels, rotation=0, show=False)
        assert np.all([label.get_rotation() == 0 for label in fig.axes[0].get_xticklabels()])

    def test_box(self):
        """
        Verify that box outputs a matplotlib figure with show=False.
        """
        box_data = np.array([[1., 2.], [3., 4.], [5., 6.]])
        labels = [f'Degree {i}' for i in [1, 2]]
        show = False

        fig = plot.box(box_data, labels, show=show)
        assert isinstance(fig, matplotlib.figure.Figure)

        fig = plot.box(box_data, labels, xlabel='xlabel', ylabel='ylabel', title='title',
                       show_zero=False, show=show)
        assert isinstance(fig, matplotlib.figure.Figure)

        # The x-tick labels are rotated 90 degrees by default and as requested otherwise.
        assert np.all([label.get_rotation() == 90 for label in fig.axes[0].get_xticklabels()])
        fig = plot.box(box_data, labels, rotation=45, show=show)
        assert np.all([label.get_rotation() == 45 for label in fig.axes[0].get_xticklabels()])

        # A constant is drawn as a horizontal line, with a legend when it is labelled.
        fig = plot.box(box_data, labels, hline=2.5, hline_label='Intercept', show=show)
        ax = fig.axes[0]
        hlines = []
        for line in ax.get_lines():
            # The y-data is a list for horizontal lines and an array for the parts of the boxes.
            # The outlier lines of the boxes hold no data at all.
            y_data = np.asarray(line.get_ydata())
            if (y_data.size > 0) and np.all(y_data == 2.5):
                hlines.append(line)
        assert len(hlines) == 1
        assert hlines[0].get_color() == 'tab:orange'
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ['Intercept']

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
