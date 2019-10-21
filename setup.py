from setuptools import setup, find_packages
# from Cython.Build import cythonize
# import numpy

setup(
    name="chmd",
    version="0.0.0",
    install_requires=["numpy", "chainer"],
    packages=find_packages(),
    # ext_modules=cythonize("dualgrad/*.pyx", numpy.get_include())
)
