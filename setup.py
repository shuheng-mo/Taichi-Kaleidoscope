# This project does not involve a package actually
# Just in case you want to install the source code as a package we formulate a template here
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name= 'utils',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Common util functions used in Taichi",
      long_description="",
      packages=['utils'])