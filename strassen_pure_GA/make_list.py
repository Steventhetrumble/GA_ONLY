import numpy as np
from Item import item
from create_options import *

def create_list2(matrix,density,mirror):
    options = find_options(matrix,mirror)
    total_size = 3**(matrix**2)

    possibilities = []

    for row in range(0, len(options), 1):
        if np.count_nonzero(options[row])> density:
            continue
        for row2 in range(0, len(options), 1):
            if np.count_nonzero(options[row2])> density:
                continue
            temporary = []
            for col in range(0, len(options[0]), 1):
                for col2 in range(0, len(options[0]), 1):
                    temporary.append([options[row][col] * options[row2][col2]])
            new_item = item(len(options),row,row2, np.array(temporary))
            possibilities.append(new_item)
    return np.array(possibilities), np.array(options)

def create_sols2(twobytwo):
    if twobytwo:
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

        final_sol = np.concatenate((C1,C2,C3,C4), axis = 1)
        return final_sol
    else:
        c1 = np.array([[1],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[1],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[1],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0]])

        c2 = np.array( [[0],[1],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[1],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[1],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0]])

        c3 = np.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[1],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[1],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0]])

        c4 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [1],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[1],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[1],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0]])


        c5 = np.array( [[0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[1],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[1],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[1],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0]])

        c6 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[1],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[1],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[1],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0]])


        c7 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [1],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[1],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[1],[0],[0]])


        c8 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[1],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[1],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[1],[0]])

        c9 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[0],
            [0],[0],[1],[0],[0],[0],[0],[0],[0],
            [0],[0],[0],[0],[0],[1],[0],[0],[0],
            [0],[0],[0],[0],[0],[0],[0],[0],[1]])


        final_sol = np.concatenate((c1,c2,c3,c4,c5,c6,c7,c8,c9), axis = 1)
        return final_sol

if __name__ == "__main__":
    print len(create_list2(2,2,False)[0])
    print create_sols2(True)