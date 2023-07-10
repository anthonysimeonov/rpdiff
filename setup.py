import os

from setuptools import find_packages
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# get the numpy include dir
numpy_include_dir = numpy.get_include()

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'rpdiff.utils.mesh_util.triangle_hash',
    sources=[
	'src/rpdiff/utils/mesh_util/triangle_hash.pyx',
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir],
    language='c++'
)

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dir_path, filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


packages = find_packages('src')
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'rpdiff' or p.startswith('rpdiff.')


def pkg_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('../..', path, filename))
    return paths


extra_pkg_files = pkg_files('src/rpdiff/descriptions')

# Gather all extension modules
ext_modules = [
    triangle_hash_module
]

setup(
    name='rpdiff',
    author='Anthony Simeonov',
    license='MIT',
    packages=packages,
    package_dir={'': 'src'},
    package_data={
        'rpdiff': extra_pkg_files,
    },
    install_requires=read_requirements_file('requirements.txt'),
    ext_modules=cythonize(ext_modules),
)

