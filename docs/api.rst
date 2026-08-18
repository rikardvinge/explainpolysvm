API reference
=============

ExPSVM
------

.. autoclass:: explainpolysvm.expsvm.ExPSVM
   :members:
   :undoc-members:

Interaction utilities
---------------------

.. autoclass:: explainpolysvm.expsvm.InteractionUtils
   :members:

.. autofunction:: explainpolysvm.expsvm.dict2array

Plotting
--------

The plotting helpers are backend-agnostic in the sense that they take bar heights or widths and
labels, and return a :class:`matplotlib.figure.Figure` when called with :code:`show=False`. The
:code:`ExPSVM.plot_*` methods prepare the data and forward keyword arguments to them.

.. autofunction:: explainpolysvm.plot.bar

.. autofunction:: explainpolysvm.plot.waterfall
