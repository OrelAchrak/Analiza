def init_mat():
    matA = []

    # Ask the user for the number of lists
    n = int(input("Enter the number of rows: "))

    # Loop over the range n
    for i in range(n):
        # Ask the user for the elements of the list, split by space
        lst = [int(item) for item in
               input("Enter the elements of row " + str(i + 1) + " separated by space: ").split()]

        # Append the list to the list of lists
        matA.append(lst)
    return matA
def display_mat(mat):
    print("the matrix is: ")
    for i in range(len(mat)):
        print(mat[i])

def sum_mat(matA, matB):
    if len(matA) != len(matB) or len(matA[0]) != len(matB[0]):
        print("Matrices cannot be added")
        return
    matC = [[0 for i in range(len(matB[0]))] for j in range(len(matA))]
    for i in range(len(matA)):
        for j in range(len(matB)):
            matC[i][j]=matA[i][j]+matB[i][j]
    return matC
def mult_mat(matA,matB):
    if len(matA[0]) != len(matB):
        print("Matrices cannot be multiplied")
        return
    matC = [[0 for i in range(len(matB[0]))] for j in range(len(matA))]
    for i in range(len(matA)):
        for j in range(len(matB)):
            for k in range(len(matB)):
                matC[i][j]+=matA[i][k]*matB[k][j]
    return matC



matA=init_mat()
matB=init_mat()
matC=sum_mat(matA,matB)
matD=mult_mat(matA,matB)

display_mat(matC)
display_mat(matD)


print("matala 1 is on git hub!!!")
