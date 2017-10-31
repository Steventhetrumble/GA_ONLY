import tables
import numpy as np
import os.path

def check_and_write(array, filename, NUM_ENTRIES):
    if not os.path.isfile(filename):
        f = tables.open_file(filename, mode='w')
        atom = tables.Float64Atom()
        array_c = f.create_earray(f.root, 'data', atom, (0,2))
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
if __name__ == "__main__":

    filename = 'outarray.h5'
    NUM_ENTRIES = 7
    x = np.array([[10, 9], [13, 13], [0, 5], [7, 7], [3, 15], [4, 1], [14, 2]])
    y = np.array([[13, 12], [14, 1], [7, 6], [8, 8], [2, 5], [1, 15], [4, 2]])
    z = np.array([[14, 1], [7, 6], [13, 12], [8, 8], [2, 5], [1, 15], [4, 2]])
    check_and_write(x, 'newfile.h5',7)
    check_and_write(y, 'newfile.h5',7)
    check_and_write(z, 'newfile.h5',7)