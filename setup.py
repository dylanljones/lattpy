from setuptools import setup

setup(
    name='lattpy',
    version='0.2.2',
    packages=['lattpy'],
    url='',
    license='MIT',
    author='Dylan Jones',
    author_email='',
    description='Python package for modeling bravais lattices',
    install_requires=['numpy>=1.18.1', 'matplotlib>=3.0.1'],
    python_requires='>=3.6'
)
