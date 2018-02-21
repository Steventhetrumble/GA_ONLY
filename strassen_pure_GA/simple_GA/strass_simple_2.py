import numpy as np
from check_unique import check_and_write


def create_Chromosome(D,mult):
    one = 0b100
    zero = 0b010
    neg_one = 0b001
    # is not rows now- is in the form [a1 a2 a3 a4] [ b1 b2 b3 b4 ]
    rows = D**3
    cols = mult
    chromosome = 0b0
    for i in xrange(rows*cols):
        choice = np.random.randint(0,34)
        chromosome = chromosome << 3
        if choice < 6:
            chromosome= chromosome|one
        elif choice < 29:
            chromosome = chromosome | zero
        else:
            chromosome = chromosome | neg_one
    return chromosome


def find_options(matrix_size, mirror):
    options_size = matrix_size ** 2
    options = []
    if mirror:
        for i in range(int((3 ** options_size))):
            rows = []
            for j in range(0, options_size):
                rows.append(1 - int(i % 3 ** (options_size - j) / (3 ** (options_size - (j + 1)))))
            number = np.count_nonzero(rows)
            if number < 5 and number > 0:
                options.append(rows)
        return options
    else:
        for i in range(int((3 ** options_size) / 2)):
            rows = []
            for j in range(0, options_size):
                rows.append(1 - int(i % 3 ** (options_size - j) / (3 ** (options_size - (j + 1)))))
            number = np.count_nonzero(rows)
            if number < 5 and number > 0:
                options.append(rows)
        return options


def determine_fitness(value, solution):
    #solution = create_sols2()
    a = np.dot(value, value.T)
    b = np.linalg.pinv(a)
    c = np.dot(value.T, b)
    d = np.dot(value.T, solution)
    e = np.dot(c.T, d)
    f = np.subtract(e, solution)
    g = np.dot(f, f.T)
    h = np.trace(g)
    return 1/(1+h), d


def final_search(D,value,final_value,x, fitness,option, solution):
    best_cost = fitness
    best_val = value
    best_final_value = final_value
    best_x = x
    # option = find_options(2,False)
    count = 0
    Column_choice = np.arange(len(value[0]))
    np.random.shuffle(Column_choice)
    for column in Column_choice:
        for j in range(0, len(option), 1):
            for k in range(0,len(option),1):
                val1 = np.copy(value)
                option1 = option[j]
                option2 = option[k]
                options = np.concatenate((option1,option2), axis= 0)
                val1[:,column]= options.T
                final_val1 = expand(D, len(value[0]), val1)
                cost1, x1 = determine_fitness(final_val1, solution)
                #print count
                count += 1
                if cost1 > best_cost:
                    best_cost = cost1
                    best_val = val1
                    best_final_value = final_val1
                    best_x = x1

                    return best_val, best_final_value, best_x, best_cost
    return best_val, best_final_value, best_x, best_cost


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


def decode(D,mult,bins):
    binary = bins
    rows = D**3
    cols = mult
    value = []
    for i in range(rows):
        temp = []
        for j in range(cols):
            if binary&0b100 :
                temp.append(1)
            elif binary&0b010 :
                temp.append(0)
            else:
                temp.append(-1)
            binary = binary >> 3
        value.append(temp)
    return np.array(value)


def expand(D, mult, value):
    rows = D ** 3
    cols = mult
    final_value=[]
    for i in range(rows/2):
        for z in range(rows/2,rows):
            temp = []
            for j in range(cols):
                temp.append(value[i][j]*value[z][j])
            final_value.append(temp)
    return np.array(final_value)


def encode(D,mult,value):
    val = value
    rows = D**3
    cols =  mult
    bins = 0b0
    for i in range(rows):
        for j in range(cols):
            bins = bins << 3
            if val[i][j] == 1:
                bins = bins | 0b100
            elif val[i][j]== 0 :
                bins = bins | 0b010
            else:
                bins = bins | 0b001
    return bins


def local_search(D,value,final_value,x, fitness, solution):
    # TODO: randomize starting position of local search
    best_cost = fitness
    best_val = value
    best_final_value = final_value
    best_x = x
    for i in range(0,len(value),1):
        for j in range(0, len(value[0]),1):
            val1 = np.copy(value)
            val2 = np.copy(value)
            if value[i][j] == 1:
                val1[i][j] = 0
                val2[i][j] = -1
                final_val1 = expand(D,len(value[0]),val1)
                final_val2 = expand(D,len(value[0]),val2)
                cost1 , x1 = determine_fitness(final_val1,solution)
                cost2 , x2 = determine_fitness(final_val2,solution)
            elif value[i][j] == 0:
                val1[i][j] = 1
                val2[i][j] = -1
                final_val1 = expand(D, len(value[0]), val1)
                final_val2 = expand(D, len(value[0]), val2)
                cost1, x1 = determine_fitness(final_val1, solution)
                cost2, x2 = determine_fitness(final_val2, solution)
            else:
                val1[i][j] = 0
                val2[i][j] = 1
                final_val1 = expand(D, len(value[0]), val1)
                final_val2 = expand(D, len(value[0]), val2)
                cost1, x1 = determine_fitness(final_val1, solution)
                cost2, x2 = determine_fitness(final_val2, solution)
            if cost1 > cost2:
                winner_cost = cost1
                winner_val = val1
                winner_final_value = final_val1
                winner_x = x1
            else:
                winner_cost = cost2
                winner_val = val2
                winner_final_value = final_val2
                winner_x = x2
            if winner_cost > best_cost:
                best_cost = winner_cost
                best_val= winner_val
                best_final_value = winner_final_value
                best_x = winner_x

                return best_val, best_final_value,  best_x, best_cost
    return best_val, best_final_value, best_x, best_cost


def crossover(D,mult,bina,binb):
    rows = D**3
    cols =  mult
    point = np.random.randint(0,rows*cols)
    maska = 0b0
    maskb = 0b0
    for i in range(0,rows*cols):
        if i < point:
            maska = maska << 3
            maskb = maskb << 3
            maska = maska | 0b111

        else:
            maskb = maskb << 3
            maska = maska << 3
            maskb = maskb | 0b111
    child1 = (bina&maska) | (binb&maskb)
    child2 = (binb&maska) | (bina&maskb)
    return child1 , child2


def mutate(D,mult,bin,rate):
    rows = D**3
    cols = mult
    binary =  bin
    maska=0b0
    maskb=0b0
    one = 0b100
    zero = 0b010
    neg_one =0b001
    mask_one = one << 3 * (mult * (D ** 3) - 1)
    mask_zero = zero << 3 * (mult * (D ** 3) - 1)
    for i in range(rows*cols):
        choicea = np.random.randint(0,100)
        maska = maska << 3
        maskb = maskb << 3
        if choicea < rate:
            choiceb = np.random.randint(0, 66)
            if (choiceb < 22 or choiceb > 55) and not(binary&mask_one):
                maskb = maskb | one
            elif (choiceb > 21 and choiceb < 56) and not(binary&mask_zero):
                maskb = maskb | zero
            else:
                maskb = maskb | neg_one
        else:
            maska = maska | 0b111
        mask_one = mask_one >> 3
        mask_zero = mask_zero >> 3
    binary = binary&maska
    binary = binary|maskb
    return binary


class strassen_search():

    def __init__(self, num_of_pop, D, multiplications):
        self.solution = create_sols2()
        self.option = find_options(2, False)
        self.purge_rate = int(num_of_pop/2)
        self.filename = '2by2.h5'
        self.prev_best_i1 = 1000
        self.prev_best_cost1 = 1000
        self.prev_best_i2 = 1000
        self.prev_best_cost2 = 1000
        self.num_of_pop = num_of_pop
        self.D = D
        self.multiplications = multiplications
        self.mute_rate = 40
        self.cost = []
        self.pop = []
        self.best_cost = 0
        self.bestx = []
        self.x = []
        self.value = []
        self.final_value = []
        self.best_value = []
        self.final_best_value = []
        self.best_i = 0
        self.best_pop = []
        self.count = 0
        self.running = 1
        for i in xrange(num_of_pop):
            chromo = create_Chromosome(self.D, self.multiplications)
            val = decode(self.D, self.multiplications, chromo)
            final_val = expand(self.D, self.multiplications, val)
            fitThis, tempx = determine_fitness(final_val,self.solution)
            self.value.append(val)
            self.final_value.append(final_val)
            self.pop.append(chromo)
            self.cost.append(fitThis)
            self.x.append(tempx)

    def tournament_selection(self):
        while(self.running):
            best_i2= self.best_i
            pop2 = np.copy(self.pop)
            cost2 = np.copy(self.cost)
            value2= np.copy(self.value)
            final_val2 = np.copy(self.final_value)
            children = 0
            while(children < self.num_of_pop):
                a = self.tournament(self.pop,2)
                b = self.tournament(self.pop,2)
                triala,trialb =crossover(self.D, self.multiplications, pop2[a], pop2[b])
                triala = mutate(self.D, self.multiplications, triala, self.mute_rate)  # int(best_cost*100)
                trialb = mutate(self.D, self.multiplications, trialb, self.mute_rate)  # int(best_cost*100)
                vala = decode(self.D, self.multiplications, triala)
                final_vala = expand(self.D, self.multiplications, vala)
                valb = decode(self.D, self.multiplications, trialb)
                final_valb = expand(self.D, self.multiplications, valb)
                costa, tempxa = determine_fitness(final_vala, self.solution)
                costb, tempxb = determine_fitness(final_valb, self.solution)
                pop2[children]=triala
                pop2[children+1]=trialb
                cost2[children]=costa
                cost2[children+1]=costb
                value2[children]=vala
                value2[children]=valb
                final_val2[children]=final_vala
                final_val2[children+1]= final_valb
                if children == 0:
                    best_i2 = children
                if cost2[children] > cost2[best_i2]:
                    best_i2 = children
                if cost2[children+1] > cost2[best_i2]:
                    best_i2 = children +1
                children += 2
            choice = np.random.choice(self.num_of_pop,int(self.num_of_pop/2))
            if cost2[best_i2] > self.cost[self.best_i]:
                self.best_i = best_i2
                self.best_pop = pop2[best_i2]
                self.best_cost = cost2[best_i2]
                self.best_value = value2[best_i2]
                self.final_best_value= final_val2[best_i2]
            for c in choice:
                self.pop[c] = pop2[c]
                self.cost[c] = cost2[c]
                self.value[c] = value2[c]
                self.final_value[c] = final_val2[c]
            self.pop[self.best_i] = self.best_pop
            self.cost[self.best_i] = self.best_cost
            self.value[self.best_i] =self.best_value
            self.final_value[self.best_i] = self.final_best_value
            self.count += 1
            self.check_for_improvement()


    def tournament(self,pop,k):
        best = None
        for i in range(k):
            ind = np.random.randint(0,self.num_of_pop)
            if (best == None) or self.cost[ind] > self.cost[best]:
                best = ind
        return best

    def simple_search(self):
        while (self.running):
            pop2 = np.copy(self.pop)
            for i in xrange(self.num_of_pop):
                triala = 0b0
                trialb = 0b0
                while (True):
                    a = int(np.random.random() * self.num_of_pop)
                    if a != i:
                        break
                triala, trialb = crossover(self.D, self.multiplications, pop2[i], pop2[a])
                triala = mutate(self.D, self.multiplications, triala, self.mute_rate)  # int(best_cost*100)
                trialb = mutate(self.D, self.multiplications, trialb, self.mute_rate)  # int(best_cost*100)
                vala = decode(self.D, self.multiplications, triala)
                final_vala = expand(self.D, self.multiplications, vala)
                valb = decode(self.D, self.multiplications, trialb)
                final_valb = expand(self.D, self.multiplications, valb)
                costa, tempxa = determine_fitness(final_vala, self.solution)
                costb, tempxb = determine_fitness(final_valb, self.solution)
                if costa > costb:
                    winner = triala
                    better_cost = costa
                    betterx = tempxa
                    better_val = vala
                    better_final_val = final_vala
                else:
                    winner = trialb
                    better_cost = costb
                    betterx = tempxb
                    better_val = valb
                    better_final_val = final_valb
                if better_cost > self.cost[i]:
                    pop2[i] = winner
                    self.cost[i] = better_cost
                    self.x[i] = betterx
                    self.value[i] = better_val
                    self.final_value[i] = better_final_val
                    # update best cost
                if self.cost[i] > self.best_cost:
                    self.best_i = i
                    self.best_cost = self.cost[i]
                    self.bestx = self.x[i]
                    self.bestpop = self.pop[i]
                    self.best_value = self.value[i]
                    self.final_best_value = self.final_value[i]

            self.pop = np.copy(pop2)
            self.count += 1
            self.check_for_improvement()

    def check_for_improvement(self):
        if self.count % 100 == 0:
            if (self.best_i == self.prev_best_i2) and (self.best_cost == self.prev_best_cost2):
                self.purge(self.purge_rate)
                print "repeat"
            elif (self.best_i == self.prev_best_i1) and (self.best_cost == self.prev_best_cost1):
                self.prev_best_i1 = 1000
                self.prev_best_cost1 = 1000
                self.prev_best_i2 = self.best_i
                self.prev_best_cost2 = self.best_cost
                self.temp_val = self.best_value
                self.temp_cost = self.best_cost
                self.best_value, self.final_best_value, self.bestx, self.best_cost = final_search(self.D,
                                                                                                  self.best_value,
                                                                                                  self.final_best_value,
                                                                                                  self.bestx,
                                                                                                  self.best_cost,
                                                                                                  self.option,
                                                                                                  self.solution)
                self.bestpop = encode(self.D, self.multiplications, self.best_value)
                self.pop[self.best_i] = self.bestpop
                self.cost[self.best_i] = self.best_cost
                self.x[self.best_i] = self.bestx
                self.value[self.best_i] = self.best_value
                self.final_value[self.best_i] = self.final_best_value
                if self.best_cost == 1:
                    print self.best_value
                    check_and_write(self.best_value.T,self.filename,self.multiplications)

                    self.purge(1)
                print self.cost
            else:
                self.prev_best_i1 = self.best_i
                self.prev_best_cost1 = self.best_cost
                self.temp_val = self.best_value
                self.temp_cost = self.best_cost
                self.best_value, self.final_best_value, self.bestx, self.best_cost = local_search(self.D,
                                                                                                  self.best_value,
                                                                                                  self.final_best_value,
                                                                                                  self.bestx,
                                                                                                  self.best_cost,
                                                                                                  self.solution)
                self.bestpop = encode(self.D, self.multiplications, self.best_value)
                self.pop[self.best_i] = self.bestpop
                self.cost[self.best_i] = self.best_cost
                self.x[self.best_i] = self.bestx
                self.value[self.best_i] = self.best_value
                self.final_value[self.best_i] = self.final_best_value
                if self.best_cost == 1:
                    print self.best_value
                    check_and_write(self.best_value.T,self.filename,self.multiplications)
                    self.purge(1)
                print self.best_cost

    def purge(self,purge_rate):
        self.pop[self.best_i] = create_Chromosome(self.D, self.multiplications)
        self.bestpop = self.pop[self.best_i]
        self.value[self.best_i] = decode(self.D, self.multiplications, self.bestpop)
        self.final_value[self.best_i] = expand(self.D, self.multiplications, self.value[self.best_i])
        self.cost[self.best_i], self.x[self.best_i] = determine_fitness(self.final_value[self.best_i], self.solution)
        self.best_cost = self.cost[self.best_i]
        self.bestx = self.x[self.best_i]
        self.best_value = self.value[self.best_i]
        self.final_best_value = self.final_value[self.best_i]
        if purge_rate > 1:
            items_for_purge = np.random.choice(self.num_of_pop,purge_rate -1)
            for i in range(len(items_for_purge)-1):
                if (items_for_purge[i]!= self.best_i):
                    chromo = create_Chromosome(self.D, self.multiplications)
                    val = decode(self.D, self.multiplications, chromo)
                    final_val = expand(self.D, self.multiplications, val)
                    fitThis, tempx = determine_fitness(final_val, self.solution)
                    self.value[items_for_purge[i]]= val
                    self.final_value[items_for_purge[i]]=final_val
                    self.pop[items_for_purge[i]]=chromo
                    self.cost[items_for_purge[i]] = fitThis
                    self.x[items_for_purge[i]] =tempx



if __name__ == "__main__":

    first = strassen_search(4,2,7)
    first.simple_search()
