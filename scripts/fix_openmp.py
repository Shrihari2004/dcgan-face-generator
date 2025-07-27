#!/usr/bin/env python3
"""
Fix for OpenMP runtime conflict.
This script sets the environment variable to resolve the libiomp5md.dll conflict.
"""

import os
import sys

def fix_openmp_conflict():
    """
    Set environment variable to fix OpenMP runtime conflict.
    """
    # Set environment variable to allow duplicate OpenMP libraries
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("OpenMP conflict fix applied: KMP_DUPLICATE_LIB_OK=TRUE")
    print("This allows the program to continue despite multiple OpenMP runtimes.")

if __name__ == '__main__':
    fix_openmp_conflict() 