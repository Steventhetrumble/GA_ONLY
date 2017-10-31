import numpy as np
import random
from Chromosome import *
from store_unique import *
from make_list import *
from merge_sort import *



def weighted_random_choice(population):
    max = sum(chromosome.get_rank() for chromosome in population)
    pick = random.uniform(0, max)
    current = 0
    for chromosome in population:
        current += chromosome.get_rank()
        if current > pick:
            return chromosome


class Population():
    def __init__(self,size,multiplications,matrix_size,density):
        self.population = []
        self.multiplications = multiplications
        self.size = size
        self.solve, self.options = create_list2(matrix_size,density,False)
        twobytwo = True
        if matrix_size > 2:
            twobytwo = False
        self.sols = create_sols2(twobytwo)#twobytwo
        print "lists generated:/n"
        for i in range(0, self.size,1):
            self.population.append(Chromosome(multiplications, self.solve ,self.sols))
        #self.elite = self.population[self.size - 1]

    def update(self):
        for i in range(0, self.size,1):
            self.population[i].update_value()
    def re_innitialize(self):
        self.population = []
        for i in range(0, self.size,1):
            self.population.append(Chromosome(multiplications, self.solve ,self.sols))



if __name__ == "__main__":
    pop = 200
    re_innit = False
    for number in range(0,10,1):
        multi = 7
        if re_innit == False:
            J =  Population(pop,multi,2,4)
        if re_innit == True:
            J.re_innitialize()
        re_innit = False
        NFC = 0
        while not re_innit and NFC < 6000:
            for i in range(0,pop,1):
                while(True):
                    q = int(random.random()*pop)
                    if q != i:
                        break
                while(True):
                    r = int(random.random()*pop)
                    if r != q and r != i:
                        break
                #while(True):
                #    s = int(random.random()*pop)
                #    if s != r and s != q and s != i:
                #        break
                #maybe sort q,r,s
                trial = J.population[q].crossover_combine(J.population[r])#,J.population[s]
                if trial.fitness > J.population[i].fitness:
                    J.population[i] = trial

                print J.population[i].fitness
                if J.population[i] == True:
                    tester = np.array(Chrom.encode_answer())
                    print tester
                    check = check_and_write(tester, '2by2_complete.h5',multi)
                    if check:
                        with open("2by2_complete.txt","a+") as f:
                            X = J.population[i].find_X()
                            string = "A=\n %s \n  X = \n %s \n C= \n %s \n" % (J.population[i].value, X ,np.dot(J.population[i].value,X))
                            print string
                            f.write(string)
                            f.close()
                        #print "YAAAAAAY"
                        re_innit = True
                        break
                    else:
                        continue
            NFC = NFC + 1
            print NFC



