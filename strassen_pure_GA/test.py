import numpy as np 

def sort_matrix2(A,B):
    matrix = np.concatenate((A,B),axis = 1)
    col_label_list = set(range(0,len(matrix[0])))
    rows_left = set(range(0,len(matrix)))
    new_row_list = []
    for c in col_label_list:
        rows_with_nonzero = [r for r in rows_left if matrix[r][c] != 0 ]
        if rows_with_nonzero != []:
            pivot = rows_with_nonzero[0]
            rows_left.remove(pivot)
            new_row_list.append(matrix[pivot])
            for r in rows_with_nonzero[1:]:
                multiplier = matrix[r][c]/matrix[pivot][c]
                matrix[r]=[i - j*multiplier for i, j in zip(matrix[r], matrix[pivot])]

    return np.array(new_row_list)

def GaussianElimination(A, B):
    MatrixA = np.copy(A)
    MatrixB = np.copy(B)
    n = len(MatrixA)
    r = len(MatrixA[0])
    m = len(MatrixB[0])
    for k in range(0,r):
        imax = find_max_pivot(MatrixA,k)
        #if (abs(MatrixA[imax][k]< alpha)):
        #    print "darn"
        temp = MatrixA[k]
        MatrixA[k] = MatrixA[imax]
        MatrixA[imax] = temp
        temp = MatrixB[k]
        MatrixB[k] = MatrixB[imax]
        MatrixB[imax] = temp
        for i in range(k+1, n):
            c = MatrixA[i][k]/MatrixA[k][k]
            MatrixA[i][k] = 0
            for j in range(k+1,r):
                MatrixA[i][j] = MatrixA[i][j] - c*MatrixA[k][j]
            for l in range(0,m):
                MatrixB[i][l] = MatrixB[i][l] - c*MatrixB[k][l]
    return  np.concatenate((np.array(MatrixA),np.array(MatrixB)),axis = 1)

def find_max_pivot(matrixA, k):
        n = len(matrixA)
        imax = k
        max_pivot = abs(matrixA[k][k]);

        for i in range(k + 1, n):
            a = abs(matrixA[i][k])
            if (a > max_pivot):
                max_pivot = a 
                imax = i 
        return imax
sol = [[1, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 1 ,0, 0], 
          [0, 0 ,0, 0],
          [0, 0 ,0, 0], 
          [1, 0, 0, 0], 
          [0, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1]]

a = [[ 1, -1,  0, -1, -1,  1,  1],
 [ 1,  1,  1,  0,  0,  1,  1],
 [ 1,  1, -1,  1,  1, -1,  0],
 [-1,  0, -1,  1,  0,  0,  0],
 [-1,  0, -1,  1,  0,  0,  0],
 [ 0, -1, -1,  0, -1,  1,  1],
 [ 1,  0,  1, -1,  0,  0,  0],
 [ 1,  1, -1,  1,  1, -1,  0],
 [ 0, -1,  1,  0,  1, -1,  1],
 [ 0,  1,  0,  1,  0,  1,  1],
 [-1,  1, -1, -1,  0, -1,  1],
 [ 0,  0,  0,  0,  0,  0,  0],
 [ 1,  0,  1, -1,  0,  0,  0],
 [-1, -1,  0,  1,  1, -1,  1],
 [ 1,  0,  1,  1, -1, -1,  0],
 [-1,  1, -1, -1,  0, -1,  1]]

x =[[ 1,  2, -1, -2],
 [-2,  2, -2,  2],
 [-1, -2,  1, -2],
 [-1,  2,  1, -2],
 [-2,  2,  2,  0],
 [ 2, -2, -2, -2],
 [ 2,  0,  2,  2]]

print sort_matrix2(a,sol)
print GaussianElimination(a,sol)
B =  np.dot(a,x)

print B