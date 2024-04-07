import numpy as np

def print_matrix(matrix):
    for row in matrix:
        for element in row:
            print(element, end=" ")  # Print each element in the row
        print()  # Move to the next row
    print()


def MaxNorm(matrix):
    """
    Function for calculating the max-norm of a matrix
    :param matrix: Matrix nxn
    :return:max-norm of a matrix
    """
    max_norm = 0
    for i in range(len(matrix)):
        norm = 0
        for j in range(len(matrix)):
            # Sum of organs per line with absolute value
            norm += abs(matrix[i][j])
        # Maximum row amount
        if norm > max_norm:
            max_norm = norm

    return max_norm

#  swapping between row i to row j in the matrix
def swap_row(mat, i, j):
    N = len(mat)
    for k in range(N + 1):
        temp = mat[i][k]
        mat[i][k] = mat[j][k]
        mat[j][k] = temp


def is_diagonally_dominant(mat):
    if mat is None:
        return False

    d = np.diag(np.abs(mat))  # Find diagonal coefficients
    s = np.sum(np.abs(mat), axis=1) - d  # Find row sum without diagonal
    return np.all(d > s)


def is_square_matrix(mat):
    if mat is None:
        return False

    rows = len(mat)
    for row in mat:
        if len(row) != rows:
            return False
    return True


def reorder_dominant_diagonal(matrix):
    n = len(matrix)
    permutation = np.argsort(np.diag(matrix))[::-1]
    reordered_matrix = matrix[permutation][:, permutation]
    return reordered_matrix


def DominantDiagonalFix(matrix):
    """
    Function to change a matrix to create a dominant diagonal
    :param matrix: Matrix nxn
    :return: Change the matrix to a dominant diagonal
    """
    #Check if we have a dominant for each column
    dom = [0]*len(matrix)
    result = list()
   # Find the largest organ in a row
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (matrix[i][j] > sum(map(abs,map(int,matrix[i])))-matrix[i][j]) :
                dom[i]=j
    for i in range(len(matrix)):
        result.append([])
        # Cannot dominant diagonal
        if i not in dom:
            print("Couldn't find dominant diagonal.")
            return matrix
    # Change the matrix to a dominant diagonal
    for i,j in enumerate(dom):
        result[j]=(matrix[i])
    return result


def swap_rows_elementary_matrix(n, row1, row2):
    elementary_matrix = np.identity(n)
    elementary_matrix[[row1, row2]] = elementary_matrix[[row2, row1]]

    return np.array(elementary_matrix)


def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions are incompatible for multiplication.")

    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return np.array(result)


def row_addition_elementary_matrix(n, target_row, source_row, scalar=1.0):

    if target_row < 0 or source_row < 0 or target_row >= n or source_row >= n:
        raise ValueError("Invalid row indices.")

    if target_row == source_row:
        raise ValueError("Source and target rows cannot be the same.")

    elementary_matrix = np.identity(n)
    elementary_matrix[target_row, source_row] = scalar

    return np.array(elementary_matrix)


def scalar_multiplication_elementary_matrix(n, row_index, scalar):

    if row_index < 0 or row_index >= n:
        raise ValueError("Invalid row index.")

    if scalar == 0:
        raise ValueError("Scalar cannot be zero for row multiplication.")

    elementary_matrix = np.identity(n)
    elementary_matrix[row_index, row_index] = scalar

    return np.array(elementary_matrix)

def Determinant(matrix, mul):
    """
    Recursive function for determinant calculation
    :param matrix: Matrix nxn
    :param mul: The double number
    :return: determinant of matrix
    """
    width = len(matrix)
    # Stop Conditions
    if width == 1:
        return mul * matrix[0][0]
    else:
        sign = -1
        det = 0
        for i in range(width):
            m = []
            for j in range(1, width):
                buff = []
                for k in range(width):
                    if k != i:
                        buff.append(matrix[j][k])
                m.append(buff)
            # Change the sign of the multiply number
            sign *= -1
            #  Recursive call for determinant calculation
            det = det + mul * Determinant(m, sign * matrix[0][i])
    return det

# Partial Pivoting: Find the pivot row with the largest absolute value in the current column
def partial_pivoting(A,i,N):
    pivot_row = i
    v_max = A[pivot_row][i]
    for j in range(i + 1, N):
        if abs(A[j][i]) > v_max:
            v_max = A[j][i]
            pivot_row = j

    # if a principal diagonal element is zero,it denotes that matrix is singular,
    # and will lead to a division-by-zero later.
    if A[i][pivot_row] == 0:
        return "Singular Matrix"


    # Swap the current row with the pivot row
    if pivot_row != i:
        e_matrix = swap_rows_elementary_matrix(N, i, pivot_row)
        print(f"elementary matrix for swap between row {i} to row {pivot_row} :\n {e_matrix} \n")
        A = np.dot(e_matrix, A)
        print(f"The matrix after elementary operation :\n {A}")
        print("------------------------------------------------------------------")
def MultiplyMatrix(matrixA, matrixB):
    """
    Function for multiplying 2 matrices
    :param matrixA: Matrix nxn
    :param matrixB: Matrix nxn
    :return: Multiplication between 2 matrices
    """
    # result matrix initialized as singularity matrix
    result = [[0 for y in range(len(matrixB[0]))] for x in range(len(matrixA))]
    for i in range(len(matrixA)):
        # iterate through columns of Y
        for j in range(len(matrixB[0])):
            # iterate through rows of Y
            for k in range(len(matrixB)):
                result[i][j] += matrixA[i][k] * matrixB[k][j]
    return result


def MakeIMatrix(cols, rows):
    # Initialize a identity matrix
    return [[1 if x == y else 0 for y in range(cols)] for x in range(rows)]
def MulMatrixVector(InversedMat, b_vector):
    """
    Function for multiplying a vector matrix
    :param InversedMat: Matrix nxn
    :param b_vector: Vector n
    :return: Result vector
    """
    result = []
    # Initialize the x vector
    for i in range(len(b_vector)):
        result.append([])
        result[i].append(0)
    # Multiplication of inverse matrix in the result vector
    for i in range(len(InversedMat)):
        for k in range(len(b_vector)):
            result[i][0] += InversedMat[i][k] * b_vector[k][0]
    return result

def RowXchageZero(matrix,vector):
    """
      Function for replacing rows with both a matrix and a vector
      :param matrix: Matrix nxn
      :param vector: Vector n
      :return: Replace rows after a pivoting process
      """

    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            # The pivot member is not zero
            if matrix[i][i] == 0:
                temp = matrix[j]
                temp_b = vector[j]
                matrix[j] = matrix[i]
                vector[j] = vector[i]
                matrix[i] = temp
                vector[i] = temp_b

    return [matrix, vector]

def Cond(matrix, invert):
    """
    :param matrix: Matrix nxn
    :param invert: Inverted matrix
    :return: CondA = ||A|| * ||A(-1)||
    """
    print("|| A ||max = ", MaxNorm(matrix))
    print("|| A(-1) ||max = ", MaxNorm(invert))
    return MaxNorm(matrix)*MaxNorm(invert)

def InverseMatrix(matrix,vector):
    """
    Function for calculating an inverse matrix
    :param matrix:  Matrix nxn
    :return: Inverse matrix
    """
    # Unveri reversible matrix
    if Determinant(matrix, 1) == 0:
        print("Error,Singular Matrix\n")
        return
    # result matrix initialized as singularity matrix
    result = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # turn the pivot into 1 (make elementary matrix and multiply with the result matrix )
        # pivoting process
        matrix, vector = RowXchange(matrix, vector)
        elementary = MakeIMatrix(len(matrix[0]), len(matrix))
        elementary[i][i] = 1/matrix[i][i]
        result = MultiplyMatrix(elementary, result)
        matrix = MultiplyMatrix(elementary, matrix)
        # make elementary loop to iterate for each row and subtracrt the number below (specific) pivot to zero  (make
        # elementary matrix and multiply with the result matrix )
        for j in range(i+1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)


    # after finishing with the lower part of the matrix subtract the numbers above the pivot with elementary for loop
    # (make elementary matrix and multiply with the result matrix )
    for i in range(len(matrix[0])-1, 0, -1):
        for j in range(i-1, -1, -1):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)

    return result


def RowXchange(matrix, vector):
    """
    Function for replacing rows with both a matrix and a vector
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Replace rows after a pivoting process
    """

    for i in range(len(matrix)):
        max = abs(matrix[i][i])
        for j in range(i, len(matrix)):
            # The pivot member is the maximum in each column
            if abs(matrix[j][i]) > max:
                temp = matrix[j]
                temp_b = vector[j]
                matrix[j] = matrix[i]
                vector[j] = vector[i]
                matrix[i] = temp
                vector[i] = temp_b
                max = abs(matrix[i][i])

    return [matrix, vector]