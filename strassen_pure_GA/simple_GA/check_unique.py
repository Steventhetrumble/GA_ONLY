import tables
import numpy as np
import os.path

def check_and_write(array, filename, NUM_ENTRIES):
    if not os.path.isfile(filename):
        f = tables.open_file(filename, mode='w')
        atom = tables.Float64Atom()
        array_c = f.create_earray(f.root, 'data', atom, (0,8))
        print array
        array_c.append(array)
        f.close()
        print "new file"
        return True
    f = tables.open_file(filename, mode='r')
    i = 0
    check = False
    for item in f.root.data[0]:
        c = np.array((f.root.data[i:i+NUM_ENTRIES,0:]))
        idx = np.where(abs((c[:,np.newaxis,:] - array)).sum(axis=2)==0)
        i = i + NUM_ENTRIES
        if len(idx[0]) == NUM_ENTRIES:
            check =True
            break
    f.close()


    if check:
        print "Duplicate Solution"
        return False
    else:
        print "Unique Solution"
        f = tables.open_file(filename, mode='a')
        f.root.data.append(array)
        f.close()
        return True

def check_and_print(filename,NUM_ENTRIES):
    array = []
    i =0
    count = 0
    f = tables.open_file(filename, mode='r')
    for item in f.root.data:

        c = np.array((f.root.data[i:i + NUM_ENTRIES, 0:]))
        if len(c) == 0:
            break
        i = i + NUM_ENTRIES
        array.append(c)

    f.close()
    return array
if __name__ == "__main__":
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

    def expand(D, mult, value):
        rows = D ** 3
        cols = mult
        final_value = []
        for i in range(rows / 2):
            for z in range(rows / 2, rows):
                temp = []
                for j in range(cols):
                    temp.append(value[i][j] * value[z][j])
                final_value.append(temp)
        return np.array(final_value)


    def determine_fitness(value, solution):
        # solution = create_sols2()
        a = np.dot(value, value.T)
        b = np.linalg.pinv(a)
        c = np.dot(value.T, b)
        d = np.dot(value.T, solution)
        e = np.dot(c.T, d)
        f = np.subtract(e, solution)
        g = np.dot(f, f.T)
        h = np.trace(g)
        return 1 / (1 + h), d
    count = 0
    final_sol = create_sols2()
    filename = '2by2.h5'
    array = check_and_print(filename,7)
    for item in array:
        count += 1
        print item
        temp = expand(2,7,item.T)
        print temp
        a, b = determine_fitness(temp,final_sol)
        print a
        print count