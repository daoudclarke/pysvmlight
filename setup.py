from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from disttest import test

ext_modules = [
    Extension("svmlight",
              ["src/svmlight.pyx", "lib/svm_common.c"],
              include_dirs = ["lib"])]

setup(
  name = 'Hello world app',
  cmdclass = {'build_ext': build_ext,
              'test' : test},
  ext_modules = ext_modules,
  options = {'test' : {'test_dir':['test']}}
)
