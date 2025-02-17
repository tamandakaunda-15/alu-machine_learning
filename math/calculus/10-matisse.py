#!/usr/bin/env python3

"""
This module contains functions for polynomial operations such as integration and differentiation.
"""

def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial represented by a list of coefficients.

    Args:
        poly (list): A list of coefficients representing the polynomial.
        C (int, optional): The constant of integration. Default is 0.

    Returns:
        list: The coefficients of the integral polynomial.
    """

    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly) \
            or not isinstance(C, (int, float)):
        return None

    integral = [C]

    for power, coef in enumerate(poly):
        if coef != 0:
            integral.append(coef / (power + 1))

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral


def poly_derivative(poly):
    """Calculates the derivative of a polynomial represented by a list of coefficients.

    Args:
        poly (list): A list of coefficients representing the polynomial.

    Returns:
        list: The coefficients of the derivative polynomial. Returns [0] if the derivative is 0.
    """

    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None

    derivative = []

    for power, coef in enumerate(poly):
        if power > 0:
            derivative.append(coef * power)

    if not derivative:
        return [0]

    return derivative
