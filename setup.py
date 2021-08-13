from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("complementaryfilter",
                  sources=["complementaryfilter.cpp"],
                  include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
                  extra_compile_args=['-std=c++17'])

c_ext.language = 'c++'
setup(
    name='complementaryfilter',
    version='1.1',
    description='Complementary filter described in https://www.cedricscheerlinck.com/continuous-time-intensity-estimation',
    ext_modules=[c_ext],
)