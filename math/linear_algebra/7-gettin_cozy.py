#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    if axis == 0:
        # Check if the number of columns are the same
        if len(mat1[0]) == len(mat2[0]):
            return mat1 + mat2
    elif axis == 1:
        # Check if the number of rows are the same
        if len(mat1) == len(mat2):
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    return None
