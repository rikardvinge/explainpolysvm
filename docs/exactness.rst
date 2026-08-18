Exactness
=========

ExplainPolySVM by default does not approximate the model it explains. This page states when that holds
and when it stops holding.

The transformation is exact
---------------------------

:code:`transform_svm()` produces a linear model with full fidelity to the original SVM. For every
observation,

.. code-block:: python

   es.decision_function(x)          # ExplainPolySVM
   svm_model.decision_function(x)   # scikit-learn

agree to numerical precision. The package's test suite asserts this exactly, for both :code:`SVC` and
:code:`SVR` models, with a tolerance of :math:`10^{-10}`. No truncation, no Taylor expansion and no
restriction on the degree of the kernel is involved: the polynomial kernel is expanded in full, and
the redundancy between permutation-equivalent interactions is what makes that affordable.

Differences at the level of :math:`10^{-12}` or so between the two decision functions are ordinary
floating-point accumulation, not a modelling error.

Masking makes the model deliberately approximate
------------------------------------------------

Masking suppresses interactions, so a masked model no longer reproduces the original decision
function. This is also the point of the masked model. The question a mask answers is *how much of the model do I lose
by keeping only these interactions*, at the cost that the exactness above no longer holds.

There are two distinct ways to mask, and they differ in whether the loss is recoverable:

* :code:`decision_function(x, mask=True)` applies the mask to that evaluation only. The full linear
  model is retained, so you can compare masked and unmasked predictions freely.
* :code:`transform_svm(mask=True)` shrinks the stored linear model permanently. The
  :code:`linear_model_is_masked` flag records this, and every subsequent call works on the reduced
  model. Recovering the discarded interactions requires transforming again from the support vectors.

For evaluating a feature selection, the first form is usually what you want.

Zero-weight interactions are not always zero contributions
----------------------------------------------------------

With :code:`coef0=0`, scikit-learn's default, every interaction below the kernel degree has an
exactly zero weight, as described in :doc:`interpretation`. Those interactions still appear in
:code:`get_interactions()` and in the linear model; they simply contribute nothing. A mask built from
weight magnitudes will rank them last.
