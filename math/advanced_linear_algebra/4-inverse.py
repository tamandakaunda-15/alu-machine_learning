def determinant(matrix):
    """Calculates the determinant of a matrix."""
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for i in range(len(matrix)):
        minor_mat = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += ((-1) ** i) * matrix[0][i] * determinant(minor_mat)
    return det

def minor(matrix):
    """Calculates the minor matrix of a matrix."""
    return [[determinant([row[:j] + row[j+1:] for i, row in enumerate(matrix) if i != k])
             for j in range(len(matrix))] for k in range(len(matrix))]

def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix."""
    return [[(-1) ** (i + j) * matrix[i][j] for j in range(len(matrix))] for i in range(len(matrix))]

def transpose(matrix):
    """Transposes a matrix."""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix."""
    return transpose(cofactor(minor(matrix)))

def inverse(matrix):
    """Calculates the inverse of a matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    det = determinant(matrix)
    if det == 0:
        return None
    
    adj = adjugate(matrix)
    return [[adj[i][j] / det for j in range(len(matrix))] for i in range(len(matrix))]
