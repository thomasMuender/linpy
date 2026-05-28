from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("linpy.util", ["linpy/util.pyx"]),
    Extension("linpy.vector3", ["linpy/vector3.pyx"]),
    Extension("linpy.quaternion", ["linpy/quaternion.pyx"]),
    Extension("linpy.transform", ["linpy/transform.pyx"]),
    Extension("linpy.scene_graph", ["linpy/scene_graph.pyx"]),
]

setup(
    name='linpy',
    version='1.2.0',
    description='A python package for calculations in R3',
    author='Thomas Muender',
    author_email='thomas.muender@gmail.com',
    packages=['linpy'],
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    setup_requires=["setuptools", "wheel", "Cython"],
)