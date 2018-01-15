import random
import numpy as np
from analog2dig import *
import xlwt


def Differential_evolution(NP, CR, F, D):
    # declare variables
    history = []
    x1 = []
    x2 = [[0] * D] * NP
    cost = []
    trial = [0] * D
    best_cost = float("inf")
    # each generation makes 100 function calls
    gen_max = D * 5000
    count = 0
    # find the fitness of initial population
    for i in range(0, NP):
        points = np.random.uniform(-1, 1, D)
        cost.append(evaluate(points))
        x1.append(points)
    # run algorithm for set number of generations
    while (count < gen_max):
        # create temporary population
        for i in range(0, NP, 1):
            for j in range(0, D, 1):
                x2[i][j] = x1[i][j]
        # for each point in the population
        for i in range(0, NP):
            # set the trial member to zero
            for z in range(D):
                trial[z] = 0
            # select 3 random points not equal to the point we are currently on
            while (True):
                a = int(random.random() * NP)
                if a != i:
                    break
            while (True):
                b = int(random.random() * NP)
                if b != a and b != i:
                    break
            while (True):
                c = int(random.random() * NP)
                if c != a and c != b and c != i:
                    break
            # main operators of Differential evolution
            for j in range(0, D):
                if (random.random() < CR):
                    trial[j] = x1[c][j] + F * (x1[a][j] + x1[b][j])
                # allow some members to persist
                else:
                    trial[j] = x1[i][j]
            # evaluate fitness
            score = evaluate(trial)
            # replace member of population if fitness is better
            if (score <= cost[i]):
                x2[i] = trial
                cost[i] = score
            else:
                x2[i] = x1[i]
        # replace the population with the temporary population
        for i in range(0, NP, 1):
            for j in range(0, D, 1):
                x1[i][j] = x2[i][j]
        # update best cost
        if cost[i] < best_cost:
            best_cost = cost[i]

        # store the current best cost every generation
        history.append([count, best_cost])
        # increase generation count
        count += 1
    return history


if __name__ == '__main__':
    NP = 100
    CR = 0.9
    F = 0.4
    D = 112
    #functions = ['hcef', 'bent_cigar', 'discus', 'rosen', 'ackley', 'weierstrass', 'rastrigin', 'griewank',
    #             'katsuura']  # ['hcef','bent_cigar','dicus','rosen', 'ackley','weierstrass','rastrigin','griewank','katsuura']
    #wb = xlwt.Workbook()
    #for f in functions:
        #ws = wb.add_sheet(f)
    for run in range(0, 51):
        history = Differential_evolution(NP, CR, F, D)
        # if run == 0:
        #    for i in range(0,len(history)):
        #        for j in range(0,len(history[0])):
        #            ws.write(i,j+run*2,history[i][j])
        # else:
        #    for i in range(0,len(history)):
        #        for j in range(1,len(history[0])):
        #            ws.write(i,j+run,history[i][j])
    # wb.save(name)
    print ("run done")

        # for entry in history:
        #     graph(entry[1], "ackley", entry[0])
