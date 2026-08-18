ExplainPolySVM
==============

ExplainPolySVM turns a trained SVM with a polynomial kernel into an equivalent *linear* model over
feature interactions. The transformation is exact, so the resulting model reproduces the decision
function of the original SVM, and every interaction between the original features gets a weight that
can be inspected, ranked and masked.

The kernel is assumed to be the common polynomial kernel

.. math::

   K(x, y \mid D, r, \gamma) = \left(r + \gamma \, x^T y\right)^D,

where :math:`r` is the independent term, :math:`\gamma` the kernel coefficient and :math:`D` the
degree. Binary classification (:code:`sklearn.svm.SVC`) and regression (:code:`sklearn.svm.SVR`) are
supported.

The method is described in

   Vinge, R., Byttner, S., Lundström, J. (2025). *Expanding Polynomial Kernels for Global and Local
   Explanations of Support Vector Machines.* In: Advances in Intelligent Data Analysis XXIII (IDA
   2025), LNCS 15669, pp. 456-468. https://doi.org/10.1007/978-3-031-91398-3_34

Getting started
---------------

.. code-block:: python

   from sklearn.svm import SVC
   from explainpolysvm import ExPSVM

   svm_model = SVC(kernel='poly', degree=2, coef0=2.5, gamma='scale').fit(x_train, y_train)

   es = ExPSVM(svm_model=svm_model)
   es.transform_svm()

   # Global explanation: the weight of every interaction, largest first.
   importance, names, order = es.feature_importance(format_names=True)

   # Local explanation: the contribution of every interaction to one prediction.
   components, names = es.decision_function_components(x_test[0], output_interaction_names=True)

Installation instructions and a longer usage walkthrough are in the project README. Worked examples
live in the ``examples`` directory of the repository.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   interpretation
   scaling
   exactness
   api
