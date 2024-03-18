import numpy as np
from colors import bcolors

def get_elemetarys(matrix):
    elem_mat = []
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            if i < n:
                if matrix[i, i + 1] != 0 and matrix[i + 1, i] != 0:
                    matrix = swap_row(matrix, i, i + 1)
                    identity = swap_row(identity, i, i + 1)
                if matrix[i + 1, i + 1] == 0:
                    raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            elem_mat.append(elementary_matrix)  # Append the elementary matrix to B
            matrix = np.dot(elementary_matrix, matrix)
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                elem_mat.append(elementary_matrix)  # Append the elementary matrix to B
                matrix = np.dot(elementary_matrix, matrix)
                identity = np.dot(elementary_matrix, identity)

    # Round the elements of the identity matrix
    identity = np.round(identity, decimals=7)

    return elem_mat

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


"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""


def swap_row(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]
    return matrix


def inverse(matrix, elem_mat):
    print(bcolors.OKBLUE,
          f"=================== Finding the inverse of a non-singular matrix using elementary row operations "
          f"===================\n {matrix}\n",
          bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            if i < n:
                if matrix[i, i + 1] != 0 and matrix[i + 1, i] != 0:
                    matrix = swap_row(matrix, i, i + 1)
                    identity = swap_row(identity, i, i + 1)
                if matrix[i + 1, i + 1] == 0:
                    raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            elem_mat.append(elementary_matrix)  # Append the elementary matrix to B
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            identity = np.dot(elementary_matrix, identity)
            print(f"The matrix after elementary operation :\n {matrix}")
            print(bcolors.OKGREEN,
                  "--------------------",
                  bcolors.ENDC)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                elem_mat.append(elementary_matrix)  # Append the elementary matrix to B
                print(f"elementary matrix for R{j + 1} = R{j + 1} + ({scalar}R{i + 1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN,
                      "--------------------",
                      bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)

    # Round the elements of the identity matrix
    identity = np.round(identity, decimals=7)

    return identity, elem_mat


if __name__ == '__main__':
    i = 0
    A = np.array([[0, 2, 3],
                  [2, 0, 4],
                  [1, 4, 6]])
    B = []  # Initialize B as an empty list
    try:
        A_inverse, B = inverse(A, B)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print(
            "===================",
            bcolors.ENDC)

    except ValueError as e:
        print(str(e))
    print("Elementary matrices stored in B:")
    for elem_matrix in B:
        print(bcolors.OKGREEN, "Location " + str(i) + " :", bcolors.ENDC)
        print(bcolors.OKGREEN, "-------------------", bcolors.ENDC)
        print(elem_matrix)
        i += 1
    print("+++++")
    print(np.dot(B[0], B[7])) #print the matrix multiply that you want
