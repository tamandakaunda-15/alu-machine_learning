import numpy as np

def definiteness(matrix):
    """Determines the definiteness of a matrix."""
    # Validate input
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.size == 0:
        return None
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    
    # Check definiteness conditions
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
