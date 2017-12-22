from analog2dig import *
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from random import random

class Particle:
    def __init__(self, D):
        self.x = np.random.uniform(0, 255, size=D)#[random()*10 - 5, random()*10 - 5]
        self.p = self.x
        self.v = np.random.uniform(-abs(-10-10), abs(10+10), size=D)

def swarm(NP, C1, C2, W, D):
    particles = []
    gbest = None
    best_fit = 0
    history = []

    for i in range(NP):
        particles.append(Particle(D))

        if i == 0:
            gbest = particles[i].p
            best_fit = evaluate(gbest)
            continue
        current_fit = evaluate(particles[i].p)
        if  current_fit < best_fit:
            gbest = particles[i].p
            best_fit = current_fit

    
    for g in range(5000*D):
        for i in particles:
            for d in range(D):
                rp, rg = np.random.uniform(size=2)
                #Update velocity in the "d" dimension
                i.v = W[g]*i.v + C1*rp*(i.p[d] - i.x[d]) + C2*rg*(gbest[d] - i.x[d])
            #Update position
            i.x += i.v
            #Update particles best
            pfitness = evaluate(i.p)
            if evaluate(i.x) < pfitness:
                i.p = i.x
                if pfitness < best_fit:
                    gbest = i.p
                    best_fit = pfitness
        
        best = [gbest[0],gbest[1], best_fit]
        print "Iter:",g, best#Gbest, evaluate(Gbest, fcn)

        if g % 500 == 0:
            history.append([g, best])
    return history


if __name__ == '__main__':
    NP = 100 #Population size
    C1 = C2 =  2.05 #Learning factors
    D = 7 #Number of search dimensions
    W = np.arange(0.9, 0.4, -0.5/(5000*D)) #Linear dec inertia
    function = 'hcef'
    history = swarm(NP, C1, C2, W, D)
    print history
