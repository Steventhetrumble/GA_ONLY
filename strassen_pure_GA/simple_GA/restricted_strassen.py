import numpy as np


def create_Chromosome(D,mult,option):
    length = len(option)
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
            if number < 3 and number > 0:
                options.append(rows)
        return options
    else:
        for i in range(int((3 ** options_size) / 2)):
            rows = []
            for j in range(0, options_size):
                rows.append(1 - int(i % 3 ** (options_size - j) / (3 ** (options_size - (j + 1)))))
            number = np.count_nonzero(rows)
            if number < 3 and number > 0:
                options.append(rows)
        return options

def determine_fitness(value):
    solution = create_sols2()
    a = np.dot(value, value.T)
    b = np.linalg.pinv(a)
    c = np.dot(value.T, b)
    d = np.dot(value.T, solution)
    #print d
    e = np.dot(c.T, d)
    f = np.subtract(e, solution)
    g = np.dot(f, f.T)
    h = np.trace(g)
    return 1/(1+h), d
    #if fitness == 1:
     #   stop = True

def final_search(D,value,final_value,x, fitness):
    best_cost = fitness
    best_val = value
    best_final_value = final_value
    best_x = x
    option = find_options(2,False)
    print len(option)
    count = 0
    Column_choice = np.arange(len(value[0]))
    np.random.shuffle(Column_choice)
    for column in Column_choice:
        print column
        for j in range(0, len(option), 1):
            for k in range(0,len(option),1):
                val1 = np.copy(value)
                option1 = option[j]
                option2 = option[k]
                options = np.concatenate((option1,option2), axis= 0)
                val1[:,column]= options.T
                final_val1 = expand(D, len(value[0]), val1)
                cost1, x1 = determine_fitness(final_val1)
                #print count
                count += 1
                if cost1 > best_cost:
                    best_cost = cost1
                    best_val = val1
                    best_final_value = final_val
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
    final_value= []
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

def local_search(D,value,final_value,x, fitness):
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
                cost1 , x1 = determine_fitness(final_val1)
                cost2 , x2 = determine_fitness(final_val2)
            elif value[i][j] == 0:
                val1[i][j] = 1
                val2[i][j] = -1
                final_val1 = expand(D, len(value[0]), val1)
                final_val2 = expand(D, len(value[0]), val2)
                cost1, x1 = determine_fitness(final_val1)
                cost2, x2 = determine_fitness(final_val2)
            else:
                val1[i][j] = 0
                val2[i][j] = 1
                final_val1 = expand(D, len(value[0]), val1)
                final_val2 = expand(D, len(value[0]), val2)
                cost1, x1 = determine_fitness(final_val1)
                cost2, x2 = determine_fitness(final_val2)
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
    compare = bin
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

    #print '{0:b}'.format(binary)
    binary = binary&maska
    #print '{0:b}'.format(binary)
    binary = binary|maskb
    #print '{0:b}'.format(binary)
    return binary




if __name__ == "__main__":
    num_of_pop=100
    D = 2
    multiplications = 7
    mute_rate = 40
    cost = []
    pop = []
    best_cost = 0
    bestx = []
    history = []
    x= []
    value = []
    final_value=[]
    best_value = []
    final_best_value = []
    #gen pop
    for i in xrange(num_of_pop):
        chromo = create_Chromosome(D,multiplications)
        val = decode(D,multiplications,chromo)
        final_val = expand(D,multiplications,val)
        fitThis , tempx = determine_fitness(final_val)
        value.append(val)
        final_value.append(final_val)
        pop.append(chromo)
        cost.append(fitThis)
        x.append(tempx)

    gen_max = D*2000
    count = 0
    while(count < gen_max):
        pop2 = np.copy(pop)
        for i in xrange(num_of_pop):
            triala = 0b0
            trialb =0b0
            while (True):
                a = int(np.random.random() * num_of_pop)
                if a != i:
                    break
            triala,trialb = crossover(D,multiplications,pop2[i],pop2[a])
            triala = mutate(D,multiplications,triala,np.random.randint(0,int(100*best_cost)+1))#int(best_cost*100)
            trialb = mutate(D,multiplications,trialb,np.random.randint(0,int(100*best_cost)+1))#int(best_cost*100)
            vala = decode(D,multiplications,triala)
            final_vala = expand(D,multiplications,vala)
            valb = decode(D,multiplications,trialb)
            final_valb = expand(D,multiplications,valb)
            costa, tempxa = determine_fitness(final_vala)
            costb, tempxb = determine_fitness(final_valb)
            if costa > costb:
                winner = triala
                better_cost = costa
                betterx = tempxa
                better_val = vala
                better_final_val = final_vala
            else:
                winner = trialb
                better_cost= costb
                betterx = tempxb
                better_val = valb
                better_final_val = final_valb
            if better_cost > cost[i]:
                pop2[i] = winner
                cost[i] = better_cost
                x[i] = betterx
                value[i] = better_val
                final_value[i] = better_final_val
        pop = np.copy(pop2)

        # update best cost
        if cost[i] > best_cost:
            best_i = i
            best_cost = cost[i]
            bestx = x[i]
            bestpop = pop[i]
            # TODO: may want to store values in an array so no need to re decode
            best_value = value[i]
            final_best_value = final_value[i]


        # store the current best cost every generation
        #history.append([count, best_cost])
        # increase generation count
        count += 1

        if count < 0:
            for s in range(0,5):
                print best_value
                print bestx
                print best_cost
                temp_val = best_value
                temp_cost = best_cost
                best_value, final_best_value, bestx, best_cost = local_search(D,best_value,final_best_value, bestx, best_cost)
                # TODO: encode might be expensive, may want ot check if best_value has changed?
                bestpop= encode(D, multiplications, best_value)
                pop[best_i]= bestpop
                cost[best_i]= best_cost
                x[best_i] = bestx
                value[best_i]= best_value
                final_value[i]= final_best_value
                if best_cost == 1:
                    print best_value
                    break
                print best_cost

        if count % 25 == 0:
            print best_value
            print bestx
            print best_cost
            temp_val = best_value
            temp_cost = best_cost
            best_value, final_best_value, bestx, best_cost = final_search(D, best_value, final_best_value, bestx,
                                                                          best_cost)
            # TODO: encode might be expensive, may want ot check if best_value has changed?
            bestpop = encode(D, multiplications, best_value)
            pop[best_i] = bestpop
            cost[best_i] = best_cost
            x[best_i] = bestx
            value[best_i] = best_value
            final_value[i] = final_best_value
            if best_cost == 1:
                print best_value
                break
            print best_cost


                #print bestx
                #print best_value
                #print bestpop'''
