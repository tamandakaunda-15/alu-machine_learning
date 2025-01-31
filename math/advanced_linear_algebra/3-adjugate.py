def determinant(matrix):
    """Calculates the determinant of a square matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    if len(matrix) == 1:
        return matrix[0][0]
    
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for col in range(len(matrix)):
        sub_matrix = [row[:col] + row[col+1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)
    return det

def minor(matrix):
    """Calculates the minor matrix of a square matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    if len(matrix) == 1:
        return [[1]]
    
    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)
    return minor_matrix

def cofactor(matrix):
    """Calculates the cofactor matrix of a square matrix."""
    minor_matrix = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            minor_matrix[i][j] *= (-1) ** (i + j)
    return minor_matrix

def adjugate(matrix):
    """Calculates the adjugate of a square matrix."""
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = [[cofactor_matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix))]
    return adjugate_matrix
