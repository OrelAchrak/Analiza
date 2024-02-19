import numpy as np
from colors import bcolors
from matrix_utility import print_matrix
from inverse_matrix import get_elemetarys


def norm(mat):
    size = len(mat)
    max_row = 0
    for row in range(size):
        sum_row = 0
        for col in range(size):
            sum_row += abs(mat[row][col])
        if sum_row > max_row:
            max_row = sum_row
    return max_row


def condition_number(A):
    # Step 1: Calculate the max norm (infinity norm) of A
    norm_A = norm(A)

    # Step 2: Calculate the inverse of A
    A_inv = np.linalg.inv(A)

    # Step 3: Calculate the max norm of the inverse of A
    norm_A_inv = norm(A_inv)

    # Step 4: Compute the condition number
    cond = norm_A * norm_A_inv

    print(bcolors.OKBLUE, "A:", bcolors.ENDC)
    print_matrix(A)

    print(bcolors.OKBLUE, "inverse of A:", bcolors.ENDC)
    print_matrix(A_inv)

    print(bcolors.OKBLUE, "Max Norm of A:", bcolors.ENDC, norm_A, "\n")

    print(bcolors.OKBLUE, "max norm of the inverse of A:", bcolors.ENDC, norm_A_inv)

    return cond


if __name__ == '__main__':
    size = 3  # You can change the size as needed
    print("Enter the elements of the matrix (each row on a new line):")
    A = []
    for i in range(size):
        row = list(map(float, input().split()))
        A.append(row)
    A = np.array(A)
    B=[]
    B= get_elemetarys(A)

    print("the last 3 elmentary of Matrix A: ")
    for mat in B[-3:]:
        print(mat)
        print("-------")
    print("the norm is: ")
    print(norm(A))


    print("-----------------------------")
    print("Date: 19/02/24")
    print(
        "Group: Haim Armias - 315569061, Yehuda Baza - 208029819, Rahamim Tadela - 208189621, Orel Achrak - 318554532")
    print("Git: https://github.com/OrelAchrak/Analiza/tree/master")
    print("Name: ")




