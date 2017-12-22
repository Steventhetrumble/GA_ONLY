import numpy as np
from create_options import *
from Item import *

def evaluate(x_array):
    analog = x_array
    digital= [0]*len(x_array)
    matrix = []
    for i in range(0,len(x_array),1):
        if (x_array[i] <= 0):
            digital[i]=0
        elif (x_array[i] >= 255):
            digital[i] = 255
        else:
            digital[i] = int(round(analog[i]))
    pos1 , pos2 = create_list2(2,2)

    for i in range(0,len(digital)):
        matrix.append(pos1[digital[i]])
    val = np.array(matrix[0])
    for i in range(1, len(digital), 1):
        val = np.concatenate((val, matrix[i]), axis=1)
    fitness = determine_fitness(val)
    return fitness

def determine_fitness(value):
    solution =create_sols2()
    a = np.dot(value, value.T)
    b = np.linalg.pinv(a)
    c = np.dot(value.T, b)
    d = np.dot(value.T, solution)
    e = np.dot(c.T, d)
    f = np.subtract(e, solution)
    g = np.dot(f, f.T)
    h = np.trace(g)
    return h
    #if fitness == 1:
     #   stop = True


def create_sols2():
    C1 = np.array([[1], [0], [0], [0],
                   [0], [1], [0], [0],
                   [0], [0], [0], [0],
                   [0], [0], [0], [0]])

    C2 = np.array([[0], [0], [1], [0],
                   [0], [0], [0], [1],
                   [0], [0], [0], [0],
                   [0], [0], [0], [0]])

    C3 = np.array([[0], [0], [0], [0],
                   [0], [0], [0], [0],
                   [1], [0], [0], [0],
                   [0], [1], [0], [0]])

    C4 = np.array([[0], [0], [0], [0],
                   [0], [0], [0], [0],
                   [0], [0], [1], [0],
                   [0], [0], [0], [1]])

    final_sol = np.concatenate((C1, C2, C3, C4), axis=1)
    return final_sol

def create_list2(matrix,density):
    options = find_options(matrix)
    total_size = 3**(matrix**2)
    reduced_options = []
    possibilities = []
    for op in range(0,len(options)):
        if np.count_nonzero(options[op])> density:
            continue
        else:
            reduced_options.append(options[op])
    for row in range(0, len(reduced_options), 1):
        for row2 in range(0, len(reduced_options), 1):
            temporary = []
            for col in range(0, len(reduced_options[0]), 1):
                for col2 in range(0, len(reduced_options[0]), 1):
                    temporary.append([reduced_options[row][col] * reduced_options[row2][col2]])
            possibilities.append(temporary)
    return np.array(possibilities),np.array(reduced_options)

if __name__ == "__main__":
    pos1 ,pos2= create_list2(2,2)
    print pos1[0], pos2