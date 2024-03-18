import numpy as np
from numpy.linalg import norm

from colors import bcolors
from matrix_utility import is_diagonally_dominant


def gauss_seidel(A, b, X0, TOL=1e-16, N=200):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming gauss seidel algorithm\n')

    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()
        if k == N-5:
            TOL = 0.00000001

    print("Maximum number of iterations exceeded")
    return tuple(x)


if __name__ == '__main__':

    A = np.array([[3, -1, 1], [0, 1, -1], [1, 1, -2]])
    b = np.array([4, -1, -3])
    X0 = np.zeros_like(b)

    A2 = np.array([[4,2,0], [2,10,4], [0,4,5]])
    b2 = np.array([2,6,5])
    X02 = np.zeros_like(b2)

    A3 = np.array([[-1, 1, 3, -3, 1],
               [3, -3, -4, 2, 3],
               [2, 1, -5, -3, 5],
               [-5, -6, 4, 1, 3],
               [3, -2, -2, -3, 5]])
    b3 = np.array([3, 8, 2, 14, 6])
    X03 = np.zeros_like(b3)

    solution =gauss_seidel(A3, b3, X03)
    print(bcolors.OKBLUE,"\nApproximate solution:", solution)


