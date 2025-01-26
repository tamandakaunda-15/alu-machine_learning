#!/usr/bin/env python3

def mat_mul(mat1, mat2):
    # Check if the number of columns in mat1 equals the number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None
    
    # Initialize the result matrix with zero values
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]
    
    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    
    return result
