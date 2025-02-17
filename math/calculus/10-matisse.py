#!/usr/bin/env python3

"""This module contains functions for polynomial operations such as integration and differentiation."""

def poly_derivative(poly):
    """Calculates the derivative of a polynomial represented by a list of coefficients."""
    
    # Check if poly is a list and if it contains only integers or floats
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None
    
    # Initialize an empty list for the derivative
    derivative = []
    
    # Calculate the derivative of each term
    for power, coef in enumerate(poly):
        if power > 0:  # Ignore the constant term (derivative of a constant is 0)
            derivative.append(coef * power)
    
    # If the derivative is an empty list (i.e., the polynomial is a constant), return [0]
    if not derivative:
        return [0]
    
    return derivative
