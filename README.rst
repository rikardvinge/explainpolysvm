ExplainPolySVM
==============

Welcome to ExplainPolySVM, a python package for feature importance analysis and feature selection
for SVM models trained using the polynomial kernel

:math:`K_p(x,y|r,D,\gamma)=(r+\gamma(x^T y))^D`,

on a binary classification problem. Here :math:`x` and :math:`y` are column vectors and :math:`r`, :math:`{\gamma}`
and :math:`D` are the independent term, scale coefficient and the degree of the polynomial kernel, respectively.

To express feature importance, the trained SVM model,

:math:`f(x) = \sum_{i\in sv}\alpha_i y_i K_p (x_i,x|r,D,\gamma) + b_0`

is transformed into a compressed linear model of the form

:math:`f(x) = \sum_{d=1} ^D \left(\begin{array}{c}D\\d\end{array}\right)r^{D-d}\left(p_d\odot
\sum_{i\in sv}\alpha_i y_i\phi(x_i|d)\right)^T \phi(x|d)`.

The transformation :math:`\phi(x|d)` contains all unique degree-:math:`d` interactions of the features of :math:`x`.
For example, :math:`\phi(x|2) = [x_0 x_0, x_0 x_1, x_0 x_2, x_1 x_1, x_1 x_2, x_2 x_2]^T` for :math:`x\in \mathbb{R}^3`.
Note that interactions :math:`x_1 x_0, x_2 x_0, x_2 x_1` are not included. This is due to the symmetry of the polynomial
kernel, namely that all permutations of the order in the interaction are identical, e.g. :math:`x_2 x_0 =x_0 x_2`.
To avoid expressing all interactions, only unique are included in :math:`\phi(x|d)`. To achieve an equivalent decision
function, the interactions in :math:`\phi(x|d)` are element-wise multiplied by the number of permutations of each
interaction. The number of permutations for each interaction is stored in :math:`p_d`. For the quadratic example above
:math:`p_d = [1,2,2,1,2,1]^T`. The number of permutations is given by :math:`d!/(n_1!n_2!...n_p!)` where
:math:`p: x\in \mathbb{R}^p` and :math:`n_i` is the number of occurrences of feature :math:`i` in the interaction.
Note that :math:`n_1+n_2+...+n_p=d`. For example, the interaction :math:`x_2 x_3 x_2 x_8 x_10` has degree 5 and
feature occurrences :math:`n_2=2`, :math:`n_3=1`, :math:`n_4=1`, :math:`n_5=1`, and, thus, the number of permutations
:math:`n_2=2`is :math:`5!/(2!1!1!1!)=60`.

Feature importance is provided by simply concatenating the terms
:math:`\left(\begin{array}{c}D\\d\end{array}\right)r^{D-d}\left(p_d\odot
\sum_{i\in sv}\alpha_i y_i\phi(x_i|d)\right)`
for all :math:`d=1...D`.


Usage
------------------

**The ExPSVM module**

The main functionality is provided by the :code:`ExPSVM` module. It interacts closely with Scikit-learn's SVC support
vector machine but can also be instantiated manually. Using a pretrained Scikit-learn SVC model :code:`svc_model` as
starting-point, a transformed SVM model using :code:`ExPSVM` can be achieved by

.. code-block::

    import expsvm
    sv = svc_model.support_vectors_
    dual_coef = svc_model.dual_coef_
    intercept = svc_model.intercept_
    d = svc_model.degree
    r = svc_model.coef0
    gamma = svc_model.gamma_

    es = expsvm.ExPSVM(sv=sv, dual_coef=dual_coef, intercept=intercept, kernel_d=d, kernel_r=r, kernel_gamma=gamma)
    es.transform_svm()

Or, simply

.. code-block::

    import expsvm
    es = expsvm.ExPSVM(svc_model=svc_model, transform=True)










