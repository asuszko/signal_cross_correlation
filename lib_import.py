# -*- coding: utf-8 -*-

__all__ = [
    "copy_signal_mf",
    "cross_correlate",
]


import os
from ctypes import (c_int,
                    c_void_p,
                    c_size_t)

from numpy.ctypeslib import load_library, ndpointer
import platform


wdir = os.path.dirname( __file__ )
lib_path = os.path.abspath(os.path.join(wdir, "lib"))

## Load the DLL
if platform.system() == 'Linux':
    cu_lib = load_library("cu_cross_correlate.so", lib_path)
elif platform.system() == 'Windows':
    cu_lib = load_library("cu_cross_correlate.dll", lib_path)

# Define argtypes for all functions to import
argtype_defs = {

    "copy_signal_mf" : [ndpointer(),
                        c_size_t],

    "cross_correlate" : [c_void_p,
                         c_void_p,
                         ndpointer("i4"),
                         c_int,
                         c_void_p],
}




## Import functions from DLL
for func, argtypes in argtype_defs.items():
    locals().update({func: cu_lib[func]})
    locals()[func].argtypes = argtypes



