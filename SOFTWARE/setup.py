# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
from cx_Freeze import setup, Executable

sys.setrecursionlimit(50000)  # increase the maximum recursion depth

setup(
    name = "measure_object_size_camera_5_circle",
    version = "0.1",
    description = "description of your script",
    executables = [Executable("measure_object_size_camera_5_circle.py")]
)

