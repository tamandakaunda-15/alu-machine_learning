#!/usr/bin/env python3
def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division
    between two matrices or a matrix and a scalar.

    Args:
        mat1: The first input matrix (numpy-compatible array or scalar).
        mat2: The second input matrix or scalar.

    Returns:
        tuple: A tuple containing four elements:
            - Element-wise sum (mat1 + mat2)
            - Element-wise difference (mat1 - mat2)
            - Element-wise product (mat1 * mat2)
            - Element-wise quotient (mat1 / mat2)
    """
    elementwise_sum = mat1 + mat2
    elementwise_diff = mat1 - mat2
    elementwise_product = mat1 * mat2
    elementwise_quotient = mat1 / mat2

    return (elementwise_sum, elementwise_diff, elementwise_product, elementwise_quotient)
