#!/usr/bin/env python3
import numpy as np

def mat_mul(mat1, mat2):
    # Ensure that the number of columns in mat1 equals the number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None
    # Perform matrix multiplication using numpy's dot function
    return np.dot(mat1, mat2)
