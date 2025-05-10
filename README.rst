ExplainPolySVM
==============

Welcome to ExplainPolySVM, a python package for feature importance analysis and feature selection
for SVM models trained using the polynomial kernel

:math:`K_p(x,y|r,D,g)=(r+g(x^Ty))^D`,

on binary classification problems. Here :math:`x` and :math:`y` are column vectors and :math:`r`, :math:`g`,
and :math:`D` are the independent term, scale coefficient and the degree of the polynomial kernel, respectively.
The greek letter gamma is often used for :math:`g`.

To express feature importance, the trained SVM model is transformed into a compressed linear version of the polynomial transformation used in the polynomial kernel.

Where to get
============

The source code is currently hosted on pip and on GitHub at: https://github.com/rikardvinge/explainpolysvm

Install with pip using

.. code-block::

    pip install explainpolysvm

To install from source, use the command

.. code-block::

    pip install ./explainpolysvm

To contribute to the development, it is recommended to install the module in edit mode including the "dev" extras to get the correct
version of pytest.

.. code-block::

    pip install -e "./explainpolysvm[dev]"

Usage
=====

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

Feature importance is retrieved by

.. code-block::

    feat_importance, feat_names, sort_order = es.feature_importance()

where :code:`feat_importance`, :code:`feat_names`, and :code:`sort_order` are all Numpy ndarrays.
:code:`feat_importance` contains the importance of each feature. :code:`feat_names` contains names of the features,
details about which interaction the feature correspond to. :code:`sort_order` provides the ordering of the interactions
to reorder the interactions returned by es.get_interactions() to the same order as returned by es.feature_importance().
Feature names are returned as strings of the form :code:`i,j,k,l,...`, where :code:`i`, :code:`j`, :code:`k`, :code:`l`
are integers in the range :math:`[1,p]` where `p` is the number of features in the original space. For example, the
interaction '0,1,0,2,2' correspond to the interaction :math:`x_0^2*x_1*x_2^2`.
Alternatively, setting the flag :code:`format_names=True` returns the feature names as formatted strings that are suitable for plotting. For
example, the interaction '0,1,0,2,2' is returned as '$x_{0}^{2}$$x_{1}$$x_{2}^{2}$', or as a list of feature names if the
:code:`feature_names` argument is passed to the ExPSVM constructor.

To return formatted feature names, use

.. code-block::

    feat_importance, formatted_feat_names, sort_order = es.feature_importance(format_names=True)

Or, to format an existing feature name list

.. code-block::

    formatted_feat_names = es.format_interaction_names(unformatted_feat_names)

Feature selection can be applied based on the contributions to the decision function. Three selection rules are
currently implemented.

.. code-block::

    # Select the 10 most important features
    feature_selection = es.feature_selection(n_interactions = 10)

    # Select 60% of the features based on importance
    feature_selection = es.feature_selection(frac_interactions = 0.6)

    # Select features that sum to 99% of the sum of all feature importances
    feature_selection = es.feature_selection(frac_importance = 0.99)

**A word of caution**

Under the hood, ExPSVM calculates a compressed version of the full polynomial transformation of the polynomial kernel.
Without compression, the number of interactions in this transformation is of order :math:`O(p^d)`, where :math:`p` is
the number of features in the original space, and :math:`d` the polynomial degree of the kernel.
The compression reduces the number of interactions by keeping only one copy of each unique interaction, with a
compression ratio of :math:`d!:1`. Even so, it is not recommended to use too large :math:`p` or :math:`d`,
both because of potential memory issues but also due to the decreasing explainability in models with very large
kernel spaces.

Example usage
=============

Global feature importance
-------------------------

This example is based on the `h_stat_strong.ipynb <https://github.com/rikardvinge/explainpolysvm/blob/main/examples/h_statistics/h_stat_strong.ipynb>`_. 
A five-dimensional binary classification dataset is created with the class separating function

.. math::
    f(x) = 0.5x_0x_1 - 0.5x_0x_2 - 0.15x_3^2 + 0.1\sum_{i_1}^5x_i + \epsilon
    f(x) > -1: class 1
    f(x) < -1: class -1

where :math:`epsilon` is a normally distributed random variable with zero mean and standard deviation 0.4; introduced 
to cause the two classes to overlap.

An SVM with a cubic kernel and parameters :math:`C=1, :math:`d=3`, :math:`gamma=scale`, :math:`r=sqrt(2)` is on data 
sampled from the formula above, achieving an accuracy of 95.7% on a held-out test set.

The global explanations are given by ExplainPolySVM are shown below

.. image:: https://github.com/rikardvinge/explainpolysvm/blob/main/media/feature_importance_signed_artificial_strong_1dpd.png
    :width: 80%

The following code can be used to compute and visualize the interaction importance
.. code-block::

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC

    from explainpolysvm import expsvm

    # Fit SVM
    C = 1
    degree = 3
    gamma = 'scale'
    r = np.sqrt(2)

    # Fit SVM

    kernel = 'poly'
    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=r)
    model.fit(X_train, y_train)

    sv = model.support_vectors_
    dual_coef = np.squeeze(model.dual_coef_)
    intercept = model.intercept_[0]
    kernel_gamma = model._gamma

    # Extract feature importance
    es = expsvm.ExPSVM(sv=sv, dual_coef=dual_coef, intercept=intercept,
                    kernel_d=degree, kernel_r=r, kernel_gamma=kernel_gamma)
    es.transform_svm()

    # Plot
    es.plot_model_bar(n_features=15, magnitude=False, figsize=(8,3))

Local explanations
------------------

Following is an example of calculating local explanations and comparing them with SHAP. The
example is based on `interaction_importance_wbc.ipynb <https://github.com/rikardvinge/explainpolysvm/blob/main/examples/wisconsin_breast_cancer_dataset/interaction_importance_wbc.ipynb>`_.
In the example, a 2D SVM is trained on the Wisonsin Breast Cancer Dataset, achieving 97.3% accuracy.

By standardizing the features to zero mean and unit variance, we can calculate the global explanations after training, as shown below.

.. image:: https://github.com/rikardvinge/explainpolysvm/blob/main/media/feature_importance_signed_wbcd.png
    :width: 80%

Even though a 2D model was trained, all but one of the 30 input features are the most important in the model, while 
the quadratic interactions are less impactful. This indicates that a linear model could suffice.

Since the trained quadratic kernel SVM is mainly linear, the impact of the individual input features
can be compared with SHAP. This is shown below for an example from the negative class, with the decision function output -1.44, using the function :code:`es.plot_sample_waterfall()`

.. image:: https://github.com/rikardvinge/explainpolysvm/blob/main/media/feature_importance_single_negative_wbcd.png
    :width: 49 %
.. image:: https://github.com/rikardvinge/explainpolysvm/blob/main/media/local_shap_same_format_wbcd.png
    :width: 49 %

The two local explanations for this sample are similar both in sign and magnitude. The reason for the different
number of remaining features is that SHAP calculates the impact of the input features, including interactions with the feature,
while ExplainPolySVM calculates the impact on the individual interactions.

A note on package maintenance
=============================

So far, ExplainPolySVM is developed by a single person. No promises will be made on maintenance nor expansions of this package.
Please let me know if you are interested in continuing its development and feel free to fork or PR!

Future development
==================

Below is a non-exhaustive list of useful and interesting features to add to the module.

- Add support for general polynomial kernels. In the current state, only the standard polynomial kernel is implemented; but any arbitrary polynomial kernel is expressible in the same way as the standard kernel. The only requirement this module have is that we can express any coefficients that are multiplied to the sum of the transformed support vectors and to keep track of the number of duplicates of the interactions.
- Add support for multi-class problems.
- Add support for the RBF Kernel by truncating the corresponding power series.
- Investigate if least-square SVM, support vector regression, one-class SVM, etc. can be expressed in similar terms as done in this project for the standard SVM.

Citations
=========

If you use ExplainPolySVM in your work we would appreciate a citation. Please see the CITATION.cff, or use the following BibText

.. code-block::

    @InProceedings{vinge2025ExPSVM,
    author="Vinge, Rikard
    and Byttner, Stefan
    and Lundstr{\"o}m, Jens",
    editor="Krempl, Georg
    and Puolam{\"a}ki, Kai
    and Miliou, Ioanna",
    title="Expanding Polynomial Kernels for Global and Local Explanations of Support Vector Machines",
    booktitle="Advances in Intelligent Data Analysis XXIII",
    year="2025",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="456--468",
    abstract="Researchers and practitioners of machine learning nowadays rarely overlook the potential of using explainable AI methods to understand models and their predictions. These explainable AI methods mainly focus on the importance of individual input features. However, as important as the input features themselves, are the interactions between them. Methods such as the model-agnostic but computationally expensive Friedman's H-statistic and SHAP investigate and estimate the impact of interactions between the features. Due to computational constraints, the investigation is often limited to second-order interactions. In this paper, we present a novel, model-specific method to explain the impact of feature interactions in SVM classifiers with polynomial kernels. The method is computationally frugal and calculates the interaction importance exactly for any order of interaction. Explainability is achieved by mathematical transformation to a linear model with full fidelity to the original model. Further, we show how the model provides for both global and local explanations, and facilitates post-hoc feature selection. We demonstrate the method on two datasets; one is an artificial dataset where H-statistics requires extra care to provide useful interpretation; and one on the real-world scenario of the Wisconsin Breast Cancer dataset. Our experiments show that the method provides reasonable, easy to interpret and fast to compute explanations of the trained model.",
    isbn="978-3-031-91398-3"
    }
