import numpy as np

from numpy.linalg import norm, inv


def gaussianElimination(mat):
    N = len(mat)

    singular_flag = forward_substitution(mat)

    if singular_flag != -1:

        if mat[singular_flag][N]:
            return "Singular Matrix (Inconsistent System)"
        else:
            return "Singular Matrix (May have infinitely many solutions)"

    # if matrix is non-singular: get solution to system using backward substitution
    return calculating_results(mat)


# function for elementary operation of swapping two rows
def swap_row(mat, i, j):
    mat[i], mat[j] = mat[j], mat[i]

def print_J_matrix(matSize, i , j, m):
    size = int(matSize)
    J = np.identity(size)
    if (i or j) >= size:
        return
    J[i][j] = m
    print(J)



def forward_substitution(mat):
    N = len(mat)
    for k in range(N):

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_row = k
        v_max = mat[pivot_row][k]
        for i in range(k + 1, N):
            if abs(mat[i][k]) > v_max:
                v_max = abs(mat[i][k])
                pivot_row = i

        # if a principal diagonal element is zero,it denotes that matrix is singular,
        # and will lead to a division-by-zero later.
        if not mat[pivot_row][k]:
            return k  # Matrix is singular

        # Swap the current row with the pivot row
        if pivot_row != k:
            swap_row(mat, k, pivot_row)
        # End Partial Pivoting

        for i in range(k + 1, N):

            #  Compute the multiplier
            m = mat[i][k] / mat[k][k]

            # subtract fth multiple of corresponding kth row element
            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j] * m

            # filling lower triangular matrix with zeros
            mat[i][k] = 0


    return -1


# function to calculate the values of the unknowns
def calculating_results(mat):
    N = len(mat)
    x = np.zeros(N)  # An array to store solution

    # Start calculating from last equation up to the first
    for i in range(N - 1, -1, -1):

        x[i] = mat[i][N]

        # Initialize j to i+1 since matrix is upper triangular
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]

        x[i] = (x[i] / mat[i][i])

    return x


if __name__ == '__main__':

    choice = input("Enter choose:")
    if choice == "1":
        A_b = [[1, -1, 2, -1, -8],
               [2, -2, 3, -3, -20],
               [1, 1, 1, 0, -2],
               [1, -1, 4, 3, 4]]

    elif choice == "2":
        A_b = [[1, 2, 3, 4, 5],
               [6, 8, 10, 50, 37],
               [18, 12, 16, 14, 15],
               [16, 29, 18, 32, 20]]

    elif choice == "3":
        A_b = [[0, 0, 0, 1, 1],
               [-1, 0, 0, 0, 1],
               [0, 1, 0, 0, 1],
               [0, 0, 1, 0, 1]]

    elif choice == "4":
        A_b = [[2, -3, 4, 5, -6, 7, 20],
               [-3, 6, -5, 8, 2, -4, -15],
               [4, -5, 7, -9, 10, -8, 30],
               [1, 2, -3, 4, -5, 6, 10],
               [-2, 3, -4, 5, 6, -7, -5],
               [3, -4, 5, -6, 7, -8, 15]]

    elif choice == "5":
        A_b = [[0.913 , 0.659 , 0.254],
                [0.457, 0.330, 0.127]]

    elif choice == "6":
        A_b = [[0, 1, -1, -1],
              [3, -1, 1, 4],
              [1, 1, -2, -3]]

    if int(choice) <= 6:
        result = gaussianElimination(A_b)
        if isinstance(result, str):
            print(result)
        else:
            print("\nSolution for the system:")
            for x in result:
                print("{:.6f}".format(x))
        for i in range(len(A_b)):
            print(A_b[i])
