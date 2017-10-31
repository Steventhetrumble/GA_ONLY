import numpy as np 
import math


def find_options(matrix_size, mirror):
    options_size = matrix_size**2
    options = []
    if mirror:
        for i in range(int((3**options_size))):
            rows = []
            for j in range(0,options_size):
                rows.append(1-int(i % 3**(options_size-j)/(3**(options_size -(j+1)))))
            
            options.append(rows)
        return options
    else:
        for i in range(int((3**options_size)/2)):
            rows = []
            for j in range(0,options_size):
                rows.append(1-int(i % 3**(options_size-j)/(3**(options_size -(j+1)))))
            
            options.append(rows)
        return options

if __name__ == "__main__":
    print find_options(2,False)

    