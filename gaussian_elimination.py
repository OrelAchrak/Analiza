import numpy as np
from colors import bcolors


def gaussianElimination(mat):
    N = len(mat)

    singular_flag = forward_substitution(mat)

    if singular_flag != -1:
        if abs(mat[singular_flag][N]) < 1e-10:  # Check for singularity with a small threshold
            return "Singular Matrix (Inconsistent System)"
        else:
            return "Singular Matrix (May have infinitely many solutions)"

    # if matrix is non-singular: get solution to system using backward substitution
    return calculating_results(mat)


# function for elementary operation of swapping two rows
def swap_row(mat, i, j):
    mat[i], mat[j] = mat[j], mat[i]


def forward_substitution(mat):
    N = len(mat)
    for k in range(N):
        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_row = k
        v_max = abs(mat[pivot_row][k])
        for i in range(k + 1, N):
            if abs(mat[i][k]) > v_max:
                v_max = abs(mat[i][k])
                pivot_row = i

        # if a principal diagonal element is zero, it denotes that matrix is singular,
        # and will lead to a division-by-zero later.
        if abs(mat[pivot_row][k]) < 1e-10:  # Check for singularity with a small threshold
            return k  # Matrix is singular

        # Swap the current row with the pivot row
        if pivot_row != k:
            swap_row(mat, k, pivot_row)

        for i in range(k + 1, N):
            # Compute the multiplier
            m = mat[i][k] / mat[k][k]

            # subtract fth multiple of corresponding kth row element
            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j] * m

            # filling lower triangular matrix with zeros
            mat[i][k] = 0

    return -1


def identity_matrix_with_solution(A_b, x):
    A_b = np.zeros((3, 4))
    np.fill_diagonal(A_b, 1)
    for i in range(len(A_b)):
        A_b[i][-1] = x[i]
    return A_b


# function to calculate the values of the unknowns
def calculating_results(A_b):
    N = len(A_b)
    x = np.zeros(N)

    # Calculate the values of the unknowns
    for i in range(N - 1, -1, -1):
        x[i] = A_b[i][-1]
        for j in range(i + 1, N):
            x[i] -= A_b[i][j] * x[j]
        x[i] = x[i] / A_b[i][i]

    return identity_matrix_with_solution(A_b, x)


if __name__ == '__main__':
    try:
        num_equations = int(input("Enter the number of equations: "))
        num_unknowns = int(input("Enter the number of unknowns: "))
        A_b = []

        print(f"Enter the augmented matrix [{num_equations}x{num_unknowns + 1}]:")
        for i in range(num_equations):
            row_values = list(map(float, input(f"Enter values for equation {i + 1}: ").split()))
            if len(row_values) != num_unknowns + 1:
                raise ValueError("Invalid row length. Please enter values for each column.")
            A_b.append(row_values)

        result = gaussianElimination(A_b)
        if isinstance(result, str):
            print(result)
        else:
            print("\nSolution for the system: ")
            for i, value in enumerate(result):
                print(f"Solution for unknown {i + 1}: {value[-1]:.6f}")
        for i in range(len(A_b)):
            print(result[i])

    except ValueError as e:
        print(f"Error: {e}")
