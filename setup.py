try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import tensorly
version = tensorly.__version__


def readme():
    with open('README.rst') as f:
        return f.read()

config = {
    'name': 'iopy',
    'packages': find_packages(exclude=['doc']),
    'description': 'Easy IO in Python with minimal dependencies.',
    'long_description': readme(),
    'author': 'Jean Kossaifi',
    'author_email': 'jean.kossaifi@gmail.com',
    'version': version,
    'url': 'https://github.com/iopy/iopy',
    'download_url': 'https://github.com/iopy/iopy/tarball/' + version,
    'install_requires': ['numpy', 'scipy'],
    'scripts': [],
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],
}

setup(**config)
