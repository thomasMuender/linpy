from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("linpy.vector", ["linpy/vector.py"], include_dirs=[numpy.get_include()]),
    Extension("linpy.quaternion", ["linpy/quaternion.py"], include_dirs=[numpy.get_include()]),
    Extension("linpy.transform", ["linpy/transform.py"], include_dirs=[numpy.get_include()]),
    Extension("linpy.util", ["linpy/util.py"], include_dirs=[numpy.get_include()]),
    Extension("linpy.scene_graph", ["linpy/scene_graph.py"], include_dirs=[numpy.get_include()]),
]

setup(
    name='linpy',
    version='1.1.0',
    description='A python package for calculations in R3',
    author='Thomas Muender',
    author_email='thomas.muender@gmail.com',
    packages=['linpy'],
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    install_requires=['numpy'],
    setup_requires=['cython', 'numpy'],
)