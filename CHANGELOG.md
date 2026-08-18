# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ExPSVM.degree_contributions()`, the contribution of each degree of the polynomial kernel to the
  decision function. The contributions and the intercept sum to the decision function, and the
  calculation requires neither a call to `transform_svm()` nor memory proportional to the number of
  interactions, so it is available for models that are too large to expand.
- `ExPSVM.degree_importance()`, aggregating the degree contributions of a set of observations, and
  `ExPSVM.plot_degree_importance()`, visualizing them as a bar chart with error bars or as a box plot.
  The support vectors are used when no observations are provided.
- `plot.box()`, a box plot helper, and error bars and markers in `plot.bar()`.
- Documentation built with Sphinx: an API reference generated from the docstrings, and pages on
  interpreting the explanations, on scaling and memory, and on when the explanations are exact.
- This changelog.

### Fixed

- Interaction weights of a linear model masked in `transform_svm(mask=True)` were returned paired
  with the names of unrelated interactions by `feature_importance()`, and plotted with those names by
  `plot_model_bar()`. Selection is now resolved once for both weights and names.
- `feature_selection()` and `set_mask()` selected the wrong interactions when applied to a model
  whose linear model had already been masked.
- `feature_selection()` returned an empty mask when the requested fraction rounded down to zero
  interactions, for example `frac_interactions=0.05` with nine interactions. At least one interaction
  is now selected.
- `plot_sample_waterfall()` did not raise on an observation of the wrong dimension; it used an
  assertion that could not raise, and that is removed entirely by `python -O`. Likewise
  `plot_sample_waterfall_degree()` constructed a `ValueError` without raising it. Both now raise.

### Changed

- `get_linear_model()` applies the `d` and `interaction_strs` selection among the interactions of a
  masked linear model. It previously ignored them, which is what caused the mismatch above.
- `plot_sample_waterfall_degree()` calculates the contribution of each degree directly instead of
  summing the interactions of the linear model. The graph is unchanged, but the method no longer needs
  a transformed model and no longer grows with the number of interactions. Its `n_degree=None`
  default, documented but broken, now works.
- Remaining `str.format()` calls replaced by f-strings.

### Security

- The PyPI token is passed to twine through the environment instead of on the command line, where it
  was visible in the runner's process list.
- The CI workflow declares `permissions: contents: read` instead of inheriting the repository default
  token scope, and checks out without persisting credentials.
