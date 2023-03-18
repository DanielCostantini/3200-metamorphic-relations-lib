from setuptools import find_packages, setup

setup(
    name='m_rs',
    packages=find_packages(include=['m_rs']),
    version='0.1.0',
    description='My first Python library',
    author='Daniel Costantini',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==7.2.2'],
    test_suite='tests',
)