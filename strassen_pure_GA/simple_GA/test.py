import numpy as np

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
    return 1/(1+h), d


A =[[ 1,  0,  0,  1,  1, -1, -1],
 [ 0, -1,  1, -1,  1,  0,  1],
 [ 0,  1,  1,  1,  1,  0,  0],
 [ 0,  1,  1,  1, -1,  0,  0],
 [ 0, -1,  1, -1,  1,  0,  1],
 [ 0,  0, -1,  1,  1, -1, -1],
 [ 1,  0,  1,  0,  0,  0,  0],
 [ 0,  1,  1,  1,  1,  0,  0],
 [-1, -1, -1,  1, -1,  0,  1],
 [-1,  0, -1,  0,  0,  0,  0],
 [ 1,  0,  0, -1,  1, -1,  1],
 [-1,  0,  1,  0,  0,  0,  1],
 [ 1,  1,  0,  1, -1,  0, -1],
 [ 0, -1,  0,  1, -1,  0,  1],
 [ 0, -1,  1, -1,  1,  0,  1],
 [ 1,  0,  0, -1,  1, -1,  1]]

C1 = np.array([1,0, 0, 0,
               0, 1, 0, 0,
               0, 0, 0, 0,
               0, 0,0, 0])

C2 = np.array([0, 0, 1, 0,
               0, 0, 0, 1,
               0, 0, 0, 0,
               0, 0, 0, 0])

C3 = np.array([0, 0, 0, 0,
               0, 0, 0, 0,
               1, 0, 0, 0,
               0, 1, 0, 0])

C4 = np.array([0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1])
A = np.array(A)

x1 = np.array(np.linalg.lstsq(A,C1))
x2 = np.array(np.linalg.lstsq(A,C2))
x3 = np.array(np.linalg.lstsq(A,C3))
x4 = np.array(np.linalg.lstsq(A,C4))

print x1
#fitness, d = determine_fitness(A)
final_sol=[]
final_sol.append(x1[0].tolist())
final_sol.append(x2[0].tolist())
final_sol.append(x3[0].tolist())
final_sol.append(x4[0].tolist())
final_sol= np.array(final_sol)
rounded = np.round(final_sol.T,2)
doty = np.dot(A,final_sol.T)
dotx = np.dot(A,rounded)
print np.round(doty,10)
print rounded

