import numpy as np
import math


def create_Chromosome(multiplication, option):
    length = int(math.log(len(option),2)*2*multiplication)
    # is not rows now- is in the form [a1 a2 a3 a4] [ b1 b2 b3 b4 ]
    one = 0b1
    zero = 0b0
    chromosome = 0b0
    for i in xrange(length):
        choice = np.random.randint(0,2)
        chromosome = chromosome << 1
        if choice < 1:
            chromosome= chromosome|one
        else:
            chromosome = chromosome | zero
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
        print options
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

def final_search(opt,D,value,final_value,x, fitness):
    best_cost = fitness
    best_val = value
    best_final_value = final_value
    best_x = x
    option = opt
    col_mod = None
    optiona= None
    optionb = None
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
                cost1, x1 = determine_fitness(final_val1)
                count += 1
                if cost1 > best_cost:
                    best_cost = cost1
                    best_val = val1
                    best_final_value = final_val
                    best_x = x1
                    col_mod= column
                    optiona = j
                    optionb= k

                    return best_val, best_final_value, best_x, best_cost, col_mod, optiona, optionb

    return best_val, best_final_value, best_x, best_cost, col_mod, optiona, optionb

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


def decode(D, mult, bins, options):
    length = int(math.log(len(options), 2))
    binary = bins
    mask= 0b0
    for i in range(0,length):
        mask = mask << 1
        mask= 0b1|mask
    rows =  D**3
    cols = mult
    value = np.zeros((rows, cols))
    for i in range(mult):
        option1 = (binary&(mask))
        binary = binary >> length
        option2 = (binary& (mask))
        temp = np.concatenate((options[option1],options[option2]))
        value[:, i] = temp.T
        binary = binary >> length
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



def encode(bin,opts,D,mult,value,col_mod, opta, optb):
    val = value
    option = opts
    length = int(math.log(len(option),2))
    maska = 0b0
    optb = optb << length
    for i in range(0, length):
        maska = maska << 1
        maska = 0b1 | maska
    maskb= maska << length
    cols =  mult
    binary = bin
    for i in range(cols):
        if i == col_mod:
            binary = binary&~maska
            binary = binary&~maskb
            binary = binary|opta
            binary = binary|optb
            break
        else:
            maska = maska << length*2
            maskb = maskb << length*2
            opta = opta << length*2
            optb = optb <<length*2

    return binary

def crossover(D,mult,length_of_options,bina,binb):
    length = int(math.log(length_of_options, 2))
    # rows = D**3
    cols =  mult
    point = np.random.randint(0,cols*2*length)
    maska = 0b0
    maskb = 0b0
    for i in range(0,cols*2*length):
        if i < point:
            maska = maska << 1
            maskb = maskb << 1
            maska = maska | 0b1

        else:
            maskb = maskb << 1
            maska = maska << 1
            maskb = maskb | 0b1
    child1 = (bina&maska) | (binb&maskb)
    child2 = (binb&maska) | (bina&maskb)
    return child1 , child2

def mutate(D,mult,length_of_options,bin,rate):
    length = int(math.log(length_of_options, 2))
    # rows = D**3
    cols = mult
    binary =  bin
    mask=0b0
    for i in range(length*2*cols):
        choice= np.random.randint(0,100)
        if choice < rate:
            mask = mask|0b1
        else:
            mask =mask|0b0
        mask = mask << 1
    binary = binary ^ mask
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
    optionz = find_options(2,True)
    #gen pop
    for i in xrange(num_of_pop):
        chromo = create_Chromosome(multiplications,optionz)
        val = decode(D,multiplications,chromo,optionz)
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
            triala,trialb = crossover(D,multiplications,len(optionz),pop2[i],pop2[a])
            triala = mutate(D,multiplications,len(optionz),triala,15)#int(best_cost*100)
            trialb = mutate(D,multiplications,len(optionz),trialb,15)#int(best_cost*100)
            vala = decode(D,multiplications,triala,optionz)
            final_vala = expand(D,multiplications,vala)
            valb = decode(D,multiplications,trialb,optionz)
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
            print best_value
            print bestx
            print best_cost


        # store the current best cost every generation
        #history.append([count, best_cost])
        # increase generation count
        count += 1


        if count % 10 ==0:
            print best_value
            print bestx
            print best_cost
            temp_val = best_value
            temp_cost = best_cost
            best_value, final_best_value, bestx, best_cost, col_mod , opta,optb = final_search(optionz,D, best_value,
                                                                        final_best_value, bestx, best_cost)


            if col_mod:
                bestpop = encode(bestpop,optionz,D, multiplications, best_value,col_mod,opta,optb)
                pop[best_i] = bestpop
                cost[best_i] = best_cost
                x[best_i] = bestx
                value[best_i] = best_value
                final_value[i] = final_best_value
            if best_cost == 1:
                print best_value
                break
            print cost


                #print bestx
                #print best_value
                #print bestpop'''
        if count < 0:
            for r in xrange(num_of_pop):
                temp_pop = pop[r]
                temp_cost = cost[r]
                temp_x = x[r]
                temp_value = value[r]
                temp_final_value = final_value[r]

                temp_value, temp_final_value, temp_x, temp_cost, col_mod, opta, optb = final_search(optionz, D, temp_value,
                                                                                                   temp_final_value, temp_x,
                                                                                                   temp_cost)

                if col_mod:
                    pop[r] = encode(temp_pop, optionz, D, multiplications, temp_value, col_mod, opta, optb)
                    cost[r] = temp_cost
                    x[r] = temp_x
                    value[r] = temp_value
                    final_value[r] = temp_final_value
                    if temp_cost > best_cost:
                        best_i = r
                        best_cost = cost[r]
                        bestx = x[r]
                        bestpop = pop[r]
                        # TODO: may want to store values in an array so no need to re decode
                        best_value = value[r]
                        final_best_value = final_value[r]

                if best_cost == 1:
                    print best_value
                    break
            print cost