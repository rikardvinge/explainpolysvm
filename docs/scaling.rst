Scaling and memory
==================

The cost of the transformation is set by the number of unique interactions, which depends only on the
number of features :math:`p` and the kernel degree :math:`D`:

.. math::

   n_{\text{int}} = \sum_{d=1}^{D} \binom{p + d - 1}{d}.

The compression that ExplainPolySVM performs — keeping one representative of each set of
permutation-equivalent interactions — reduces the full :math:`\mathcal{O}(p^D)` expansion by a factor
:math:`D!`, but the growth in :math:`p` and :math:`D` is still high.

Interactions of the highest degree, :math:`\binom{p + D - 1}{D}`, dominate the total:

===========  =====  ====================
:math:`p`    D      interactions
===========  =====  ====================
30           2      465
30           3      4,960
50           4      292,825
100          3      171,700
100          4      4,421,275
384          2      73,920
768          2      295,296
1024         2      524,800
384          3      9,511,040
1024         3      179,481,600
===========  =====  ====================

The cost of high :math:`p` and :math:`d`
----------------------------------------

The linear model itself is one float per interaction, so a million interactions is roughly 8 MB. The
transformation is more demanding than the result: it evaluates every interaction for every support
vector, so intermediate arrays are of the order :math:`n_{\text{SV}} \times n_{\text{int}}` and are
what exhausts memory first.

:code:`transform_svm(reduce_memory=True)` loops over observations rather than transforming all of
them at once, which lowers the peak. It does not change the size of the result.

Practical guidance
------------------

* Standardise features and keep :math:`D` modest. Beyond a handful of degrees, an RBF kernel is often
  the better modelling choice, and the explanations become hard to interpret regardless.
* Estimate :math:`n_{\text{int}}` before transforming a model with many features or a high degree.
  :code:`math.comb(p + D - 1, D)` is enough to tell you whether the transformation is feasible.
* Use :code:`set_mask()` and :code:`transform_svm(mask=True)` to work with a reduced model once you
  know which interactions matter. See :doc:`exactness` for the consequences.
