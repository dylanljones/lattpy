from setuptools import setup, find_packages

setup(
    name='lattpy',
    version='0.2.3',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Dylan Jones',
    author_email='',
    description='Python package for modeling bravais lattices',
    install_requires=['numpy>=1.18.1', 'matplotlib>=3.0.1'],
    python_requires='>=3.6'
)
