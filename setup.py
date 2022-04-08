from setuptools import setup, find_packages

setup(
    name='ExplainPolySVM',
    version='0.1',
    description='Module for feature importance extraction and feature selection for Support Vector Machines trained with polynomial kernels.',
    long_description=open('README.rst').read(),
    url='https://github.com/rikvinge/expsvm',
    author='Rikard Vinge',
    author_email='rikard.vinge.github@gmail.com',
    license='BSD 3',
    packages=find_packages,
    install_requires=['numpy>=1.21.5', 'scikit-learn>=1.0.2'],
    extras_requires={
        'test': ['pytest>=6.2.5'],
        'examples': ['matplotlib']
        },
    keywords=['feature importance', 'support vector machine', 'svm', 'peature selection', 'polynomial kernel'],
    classifiers= [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ]
)