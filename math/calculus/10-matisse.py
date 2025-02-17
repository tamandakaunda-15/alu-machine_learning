def poly_integral(poly, c=0):
    if not isinstance(poly, list) or not isinstance(c, int) or not poly:
        return None
        
    result = [c]  # Start with the integration constant
    
    # Integrate each term
    for i, coef in enumerate(poly):
        # Add integrated term: divide coefficient by (power + 1)
        new_coef = coef / (i + 1)
        # Convert to integer if it's a whole number
        if new_coef.is_integer():
            new_coef = int(new_coef)
        result.append(new_coef)
        
    return result
