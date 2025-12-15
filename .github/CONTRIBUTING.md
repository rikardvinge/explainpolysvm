# Contributing

First of all, thank you for considering to contribute to this project! All forms of contributions are welcome, including but not limited to:

- Expansion to other forms of the SVM, kernels or models.
- Application-oriented improvements and extensions, such as toward contrafactual explanations.
- Documentation updates, bug fixes and code improvements.

## Workflow

0. For significant modifications or any bugs spotting, please consider opening an issue for discussion beforehand.
1. Fork and pull the latest repository (Click the **Fork** button on GitHub).
2. Clone your fork:
   ```sh
   git clone https://github.com/your-username/explainpolysvm.git
   ```
3. Navigate into the project directory
   ```sh
   cd explainpolysvm
   ```
4. Add the upstream repository
   ```sh
   git remote add upstream https://github.com/rikardvinge/explainpolysvm.git
   ```
5. Create a local branch
   ```sh
   git checkout -b feature-branch
   ```
   and use a descriptive branch name related to your changes, e.g. `feature/contrafactuals`.
6. Make your changes following the [PEP 8](https://peps.python.org/pep-0008) coding standard.
7. Write clear and concise commit messages:
   ```sh
   git commit -m 'Short description of changes' -m'and more details'
   ```
8. Push your branch to your fork
   ```sh
   git push origin feature-branch  
   ```
9. Go to your fork on GitHub and click **Compare & pull request**. Provide a detailed explanation of your changes and link to relevant issues.


# Code of Conduct
As contributor you are expected to follow the [Code of Conduct as specified by the Linux Foundation](https://docs.linuxfoundation.org/lfx/mentorship/mentor-guide/code-of-conduct).

## License
By contributing, you agree that your work will be licensed under [BSD-3](https://spdx.org/licenses/BSD-3-Clause.html).

## Need Help?
If you have any questions, feel free to open an issue or contact the project maintainers.


# Code structure 

The main code for ExplainPolySVM reside in `src/explainpolysvm`.

## expsvm

In `expsvm.py`, the main functionality for calculating interaction importance is located.

## plot

`plot` contains code for creating bespoke interaction importance plots.

1. `_bar.py`, basic par plot of the global interaction importance.
2. `_waterfall.py`, waterfall plot for local, single sample, interaction contributions.

## examples

`examples` contain example notebooks to showcase how to use the package.
