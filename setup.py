from setuptools import setup, find_packages

setup(
    name='explainpolysvm',
    version='0.4',
    description='Module for Global and Local explanations and feature selection for Support Vector Machines trained '
                'with polynomial kernels.',
    long_description=open('README.rst').read(),
    url='https://github.com/rikardvinge/explainpolysvm',
    author='Rikard Vinge',
    author_email='research@rikardvinge.se',
    license='BSD 3',
    packages=find_packages(),
    install_requires=['numpy>=1.21.5', 'scikit-learn>=1.0.2', 'matplotlib'],
    extras_require={'dev': ['pytest>=6.2.5']},
    keywords=['feature importance', 'support vector machine', 'svm', 'feature selection', 'polynomial kernel', 'xai'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ]
)