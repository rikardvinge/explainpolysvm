Interpreting the explanations
=============================

This page explains what the numbers produced by ExplainPolySVM mean and when the interpretation no longer holds.

Global explanations: interaction weights
----------------------------------------

After :code:`transform_svm()`, the model holds one weight per unique interaction.
:code:`feature_importance()` returns those weights, their names and the sorting order.

An interaction weight is **the change in the decision function caused by a unit change in one of the
features constituting the interaction, with all other features held constant**. A feature that
appears in several interactions influences the decision function through all of them, so its total
effect is the sum over every interaction it takes part in.

Two consequences follow:

* **Weights are comparable across interactions only if the features are comparably scaled.** The
  weight of :math:`x_0 x_1` and the weight of :math:`x_2^2` can be compared directly when all
  features share a range, and cannot when one feature is measured in millimetres and another in
  kilometres. Standardising the features before training is the usual remedy.
* **The sign matters.** :code:`feature_importance(magnitude=False)` returns signed weights, which
  say in which direction the interaction pushes the decision function. The default,
  :code:`magnitude=True`, ranks by absolute value and discards that information.

Interaction names are strings of feature indices: :code:`'0,0,2'` means
:math:`x_0^2 x_2`. The order of the indices is irrelevant and is provided in ascending order by the 
feature index. :code:`format_names=True` renders them for plotting, either as mathtext or, when 
:code:`feature_names` has been set, as :code:`(name_a)*(name_b)`.

Local explanations: decision function components
------------------------------------------------

:code:`decision_function_components(x)` multiplies the transformed observation element-wise by the
linear model, giving the contribution of every interaction to that particular prediction. These
contributions are **additive**: they sum, together with the intercept, to
:code:`decision_function(x)`, which is what :code:`plot_sample_waterfall()` visualises.

For classification the decision function is the signed distance to the separating hyperplane, not a
probability. For regression it is the predicted value.

How the kernel parameters shape what you can see
------------------------------------------------

The weight of a degree :math:`d` interaction carries the factor

.. math::

   \binom{D}{d} \, r^{D-d} \, \gamma^{d}.

The independent term :math:`r` therefore controls how much of the model lives in the lower degrees.
In particular, **scikit-learn's default is** :code:`coef0=0.0`, which makes :math:`r^{D-d} = 0` for
every :math:`d < D`: all lower-degree interactions have exactly zero weight and the entire model sits
in degree :math:`D`. This is correct behaviour rather than a defect, but it is worth knowing before
concluding that a model "uses only the highest-order interactions" — with :code:`coef0=0` it could
not have done anything else. Set :code:`coef0` to a positive value if you want the lower degrees to
participate.

Feature selection
-----------------

:code:`feature_selection()` ranks interactions by weight magnitude and returns a boolean mask, which
:code:`set_mask()` stores on the model. Masking suppresses interactions in the decision function of
the *already trained* model; it does not retrain anything. See :doc:`exactness` for what masking does
to the fidelity of the model.
