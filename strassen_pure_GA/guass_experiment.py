import numpy as np
alpha = 1e-10
 
def GaussianElimination(MatrixA, MatrixB):
    n = len(MatrixA)
    r = len(MatrixA[0])
    m = len(MatrixB[0])
    for k in range(0,r):
        imax = find_max_pivot(MatrixA,k)
        if (abs(MatrixA[imax][k]< alpha)):
            print "hello"
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
    return np.concatenate((np.array(MatrixA),np.array(MatrixB)),axis = 1)

def back_sub(upper_triangle):
        for i in range(7 - 1, -1, -1):
            if upper_triangle[i][i] != 1:
                upper_triangle[i][:] = upper_triangle[i][:] / upper_triangle[i][i]
            if any(upper_triangle[:][i]) != 0:
                for j in range(0, i, 1):
                    upper_triangle[j] = upper_triangle[j] - upper_triangle[i] * upper_triangle[j][i]
        
        return upper_triangle    


def find_max_pivot(matrixA, k):
    n = len(matrixA)
    imax = k
    print k,n,len(matrixA[0])
    max_pivot = abs(matrixA[k][k]);

    for i in range(k + 1, n):
        a = abs(matrixA[i][k])
        if (a > max_pivot):
            max_pivot = a 
            imax = i 
       
    return imax
if __name__ == "__main__":
    print alpha - 1
    B = [[1, 0, 0, 0],
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
    
    #A = [[ 1,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  0,  0,  0,  0,  0],
    #    [ 1,  0,  1,  0,  0,  0,  0],
    #    [ 0,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  0,  0,  0,  1,  1],
    #    [ 0,  0,  1,  1,  0,  1,  0],
    #    [ 0,  0,  0, -1,  0,  0,  1],
    #    [-1,  0,  0,  0, -1,  0,  0],
    #    [ 0,  1,  0,  0,  1,  1,  0],
    #    [-1,  0,  0,  0,  0,  1,  0],
    #    [ 0,  0,  0,  0,  0,  0,  0],
    #    [ 0 , 0,  0,  0,  0,  0,  0],
    #    [ 0,  1,  0,  0,  0,  0, -1],
    #    [ 0,  0,  0,  0,  0,  0,  0],
    #    [ 0 , 0,  0,  0,  0,  0, -1]]

    A =[[ 1,  0,  1,  1,  0,  0,  0],
 [ 0,  0, -1,  1,  0,  0,  0],
 [-1,  0,  0,  1,  0,  0,  0],
 [-1,  0,  1,  0,  0,  0,  0],
 [ 1,  0, -1,  1,  1,  0,  1],
 [ 0,  0,  1,  1, -1,  0, -1],
 [-1,  0,  0,  1, -1,  0,  0],
 [-1,  0, -1,  0,  1,  0,  0],
 [ 0,  1, -1,  1,  0,  0, -1],
 [ 0,  1,  1,  1,  0,  0,  1],
 [ 0, -1,  0,  1,  0,  1,  0],
 [ 0, -1, -1,  0,  0, -1,  0],
 [ 0,  1,  1,  0,  0,  0,  1],
 [ 0,  1, -1,  0,  0,  0, -1],
 [ 0, -1,  0,  0,  0, -1,  0],
 [ 0, -1,  1,  0,  0,  1,  0]]
    C = GaussianElimination(A,B)
    print back_sub(C)
