# -*- coding: utf-8 -*-

__all__ = [
    "copy_signal_mf",
    "cross_correlate",
]


import os
from ctypes import (c_int,
                    c_void_p,
                    c_size_t)

from numpy.ctypeslib import ndpointer


# Load the shared library
from pycu_interface import load_lib
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
cu_lib = load_lib(lib_path,"cu_cross_correlate")


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



