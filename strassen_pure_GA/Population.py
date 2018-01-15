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
            self.population.append(Chromosome(self.multiplications, self.solve ,self.sols))



if __name__ == "__main__":
    pop = 100
    re_innit = False
    for number in range(0,10,1):
        multi = 7
        if re_innit == False:
            J =  Population(pop,multi,2,2)
        if re_innit == True:
            J.re_innitialize()
        re_innit = False
        NFC = 0
        while not re_innit and NFC < 5000:
            for i in range(0,pop,1):
                #if (J.population[i].prospect):
                #    J.population[i].local_search()
                #else:
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
                    if J.population[q].fitness > J.population[r].fitness:
                        a = q
                        b = r 
                    else:
                        a = r 
                        b = q
                    trial = J.population[a].crossover_combine(J.population[b])#,J.population[s]
                    mutation = np.random.randint(0,100)
                    
                    
                    if trial.fitness > J.population[i].fitness:
                        J.population[i] = trial
                        J.population[i].update_x()
                    
                    if mutation < 100 and len(J.population[i].choices) != J.population[i].multiplications:
                        print "hello"
                        J.population[i].directed_local_search()
                    elif mutation < 100:
                        J.population[i].local_search()


                    print J.population[i].fitness
                    if J.population[i].fitness >= .2:
                        print "*******************************"
                        print J.population[i].solution
                        print J.population[i].choices
                        print J.population[i].value
                        print J.population[i].sorted
                        print J.population[i].complete 
                        print J.population[i].X
                        print "*********************************"
                    if J.population[i].stop == True:
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
            #print NFC



