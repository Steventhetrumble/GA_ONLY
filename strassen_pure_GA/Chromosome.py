import numpy as np
from Item import item
import random
#from create_list import *
import sys
import math
alpha = 1e-10
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


class Chromosome():

    def __init__(self, multiplications, list ,sols):
        self.multiplications = multiplications
        self.options = list
        self.solution = sols
        self.Chromosome = np.random.choice(self.options, multiplications)
        val = np.array(self.Chromosome[0].result)
        for i in range(1,multiplications,1):
            val = np.concatenate((val, self.Chromosome[i].result), axis = 1)
        self.value = val
        self.fitness = 0
        self.determine_fitness()
        self.stop = False
        self.encode = []
        self.sorted = []
        self.complete = []
        self.X = []
        self.choices = []
        self.update_x()
        self.prospect = False
        



    def back_sub(self,ut,choice = None):
        upper_triangle =  np.copy(ut)
        for i in range(self.multiplications - 1, -1, -1):
            if upper_triangle[i][i] != 1:
                upper_triangle[i][:] = upper_triangle[i][:] / upper_triangle[i][i]
            if any(upper_triangle[:][i]) != 0:
                for j in range(0, i, 1):
                    upper_triangle[j] = upper_triangle[j] - upper_triangle[i] * upper_triangle[j][i]
        self.complete = upper_triangle
        return upper_triangle[:self.multiplications, self.multiplications:]

    def encode_answer(self):
        for item in self.Chromosome:
            self.encode.append([item.row1,item.row2])
        return self.encode


    def update_value(self):
        val = np.array(self.Chromosome[0].result)
        for i in range(1, self.multiplications, 1):
            val = np.concatenate((val, self.Chromosome[i].result), axis=1)
        self.value = val
        self.determine_fitness()

    def update_x(self):
        self.find_X()
        self.check_X()

    def determine_fitness(self):
        #self.partition = np.concatenate((self.value, self.solution), axis=1)

        solution = self.solution
        a = np.dot(self.value, self.value.T)
        b = np.linalg.pinv(a)
        c = np.dot(self.value.T, b)
        d = np.dot(self.value.T, solution)
        e = np.dot(c.T, d)
        f = np.subtract(e, solution)
        g = np.dot(f, f.T)
        h = np.trace(g)
        i = 1 / (1 + h)
        self.fitness = i
        if self.fitness == 1:
            self.stop = True

    def find_X(self):
        X = sort_matrix2(self.value[:],self.solution[:])
        self.sorted = X
        X = self.back_sub(X)
        self.X = X

    def check_X(self):
        search_index = []
        unique_items , unique_index = np.unique(self.X, return_index = True, axis = 0)
        for row in unique_index:
            if self.X[row].any() != 0 and abs(self.X[row].any()) <= 1:
                search_index.append(row)
        
        if len(search_index) == self.multiplications:
            self.prospect = True
        else:
            self.prospect = False
        
        self.choices = np.array(search_index)#,ind

    def local_search(self):
        choice = np.random.choice(np.arange(self.multiplications))
        old_fitness = self.fitness
        old_item = self.Chromosome[choice]
        search_range = 255 #int((self.fitness)*(self.fitness)*len(self.options)) #int((self.fitness)*
        upper_range = len(self.options) - search_range
        start = int(random.random()*upper_range)
        end = start + search_range
        for i in range(start,end,1):
            self.Chromosome[choice] = self.options[i]
            self.update_value()
            if self.fitness >= old_fitness:
                old_item = self.Chromosome[choice]
                old_fitness = self.fitness
                if self.stop == True:
                    break
                #break
        self.Chromosome[choice]= old_item
        self.update_value()
        self.update_x()

    def crossover_combine(self,chrome_b):#,chrome_c
        index_a = self.choices
        index_b =chrome_b.choices
        #index_c =chrome_c.check_X()
        trial = Chromosome(self.multiplications,self.options,self.solution)
        for i in range(0,self.multiplications,1):
            if i in index_a:
                trial.Chromosome[i] = self.Chromosome[i]
            elif i in index_b:
                trial.Chromosome[i] = chrome_b.Chromosome[i]
        trial.update_value()
        return trial   

    def directed_local_search(self):
        bad_choice = set(np.arange(self.multiplications)) - set(self.choices)
        choice = np.random.choice(list(bad_choice)) 
        old_fitness = self.fitness
        old_item = self.Chromosome[choice]
        search_range = 255#int((self.fitness)*(self.fitness)*len(self.options)) #int((self.fitness)*
        upper_range = len(self.options) - search_range
        start = int(random.random()*upper_range)
        end = start + search_range
        for i in range(start,end,1):
            self.Chromosome[choice] = self.options[i]
            self.update_value()
            if self.fitness >= old_fitness:
                old_item = self.Chromosome[choice]
                old_fitness = self.fitness
                if self.stop == True:
                    break
                #break
        self.Chromosome[choice]= old_item
        self.update_value()
        self.update_x()       


if __name__ == "__main__":
    for i in range(0,10000,1):
        new_Chrom = Chromosome(7)
        print new_Chrom.value
        print new_Chrom.fitness
        print new_Chrom.find_X()
        print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"

        for i in range(0,40,1):
            print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
            new_Chrom.local_search()
            print new_Chrom.value
            print new_Chrom.fitness




