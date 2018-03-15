# -*- coding: utf-8 -*-
"""
If build is not found, it can be obtained in 
the cuda_manager repo in the shared_utils folder.
"""

import argparse
import os

from pycu_interface.shared_utils.build import build

__lib_name = "cu_cross_correlate"



def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-arch', '--arch',
                        action="store", dest="arch",
                        help="CUDA hardware architecture version",
                        default="sm_30")
    
    parser.add_argument('-cc_bin', '--cc_bin',
                        action="store", dest="cc_bin",
                        help="Path to the cl.exe bin folder on Windows",
                        default=None)
    
    args = parser.parse_args()

    module_path = os.path.abspath(os.path.dirname(__file__))
    print(module_path)
    build(module_path, __lib_name, args.arch, args.cc_bin)



if __name__ == "__main__":
    main()