from strassen_pure_GA.analogous.create_options import *
from strassen_pure_GA.analogous.create_options import *


def evaluate(x_array):
    matrix = [[0]*7]*16
    a = 7
    b = 16
    for i in range(0,b):
        for j in range(0,a):
            matrix[i][j] = int(round(x_array[i*7 + j]))
    val = np.array(matrix)
    fitness = determine_fitness(val)
    print fitness
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