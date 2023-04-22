from setuptools import find_packages, setup
from codecs import open
from os import path

# https://towardsdatascience.com/deep-dive-create-and-publish-your-first-python-library-f7f618719e14

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name='metamorphic_relations',
    version="0.1.2",
    description="Metamorphic relations library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://3200-metamorphic-relations-lib.readthedocs.io",
    author="Daniel Costantini",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(include=['metamorphic_relations']),
    include_package_data=True,
    install_requires=["numpy", "matplotlib", "scipy", "pytest", "setuptools", "scikit-learn",
                      "tensorflow", "keras", "tabulate"],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==7.2.2'],
    test_suite='tests',
)