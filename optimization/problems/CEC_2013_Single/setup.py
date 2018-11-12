from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import sys

source_files = ["cec2013single/cec2013.pyx", "cec2013single/cec2013_func.c"]

cec2013single = Extension("cec2013single.cec2013",
                          source_files,
                          libraries=["m"])  # Unix-like specific


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='cec2013single',
    version='0.1',
    author='Clodomir Santana and Daniel Molina',
    author_email='cjsj@ecomp.poli',
    maintainer='Clodomir Santana',
    description='Package for benchmark for the Real Single Objective Optimization session on IEEE Congress '
                'on Evolutionary Computation CEC\'2013',
    long_description=open('README.rst').read(),
    license='GPL V3',
    packages=['cec2013single'],
    install_requires=['cython', 'numpy'],
    ext_modules=cythonize(cec2013single),
    package_data={'cec2013single': ['cec2013_data/*.txt']},
    tests_require=['pytest'],
    cmdclass={'build_ext': build_ext, 'test': PyTest},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    ]
)
