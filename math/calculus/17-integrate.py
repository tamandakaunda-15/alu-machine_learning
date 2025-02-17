def poly_integral(poly, C=0):
    # Check if poly is a list and C is an integer
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly) or not isinstance(C, (int, float)):
        return None
    
    # Calculate the integral of the polynomial
    integral = [C]  # Start with the constant of integration
    
    for power, coef in enumerate(poly):
        if coef != 0:  # Only process non-zero coefficients
            integral.append(coef / (power + 1))  # Divide by the new power (power + 1)
    
    # Remove any trailing zeros from the integral (if they exist)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
