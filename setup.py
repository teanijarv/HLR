#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['scipy>=1.7.3',
                'pandas>=1.3.5',
                'matplotlib>=3.5.2',
                'seaborn>=0.11',
                'statsmodels>=0.13',
                'pingouin>=0.5.3']
test_requirements = ['pytest>=3']

setup(
    author="Toomas Erik Anijärv",
    author_email='toomaserikanijarv@gmail.com',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="HLR - Hierarchical Linear Regression for Python",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description_content_type='text/markdown',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='HLR',
    name='HLR',
    packages=find_packages(include=['HLR', 'HLR.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/teanijarv/HLR',
    version='0.2.1',
    zip_safe=False,
)
