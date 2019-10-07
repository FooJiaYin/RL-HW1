import numpy as np
import math
import environment

def initProb(map): # Initialize policy with all equal probability
    prob = np.empty((map.shape[0], map.shape[1], 4))
    #print(map)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] == 0:
                prob[i][j].fill(0.0)
            else:
                prob[i][j].fill(0.25)
                if i == 0:              # first row
                    prob[i][j] = [0.0, 1.0/3, 1.0/3, 1.0/3]
                if i == map.shape[0]-1: # last row
                    prob[i][j] = [1.0/3, 0.0, 1.0/3, 1.0/3]
                if j == 0:              # first col
                    prob[i][j] = [1.0/3, 1.0/3, 0.0, 1.0/3]
                if j == map.shape[1]-1: # last col
                    prob[i][j] = [1.0/3, 1.0/3, 1.0/3, 0.0]

    if map[0][0] != 0:
        prob[0][0] = [0, 0.5, 0, 0.5]
    if map[0][map.shape[1]-1] != 0:
        prob[0][map.shape[1]-1] = [0, 0.5, 0.5, 0]
    if map[map.shape[0]-1][0] != 0:
        prob[map.shape[0]-1][0] = [0.5, 0, 0, 0.5]
    if map[map.shape[0]-1][map.shape[1]-1] != 0:
        prob[map.shape[0]-1][map.shape[1]-1] = [0.5, 0, 0.5, 0]

    return prob

def evalPolicy(map, value, prob):
    new_value = np.copy(value)
    error = 0
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            new_value[i][j] = map[i][j]
            if prob[i][j][0] > 0.000001:
                new_value[i][j] += prob[i][j][0] * value[i-1][j]
            if prob[i][j][1] > 0.000001:
                new_value[i][j] += prob[i][j][1] * value[i+1][j]
            if prob[i][j][2] > 0.000001:
                new_value[i][j] += prob[i][j][2] * value[i][j-1]
            if prob[i][j][3] > 0.000001:
                new_value[i][j] += prob[i][j][3] * value[i][j+1]
            error += value[i][j] - new_value[i][j]
            #print(value[i][j], new_value[i][j])
    #print(new_value)
    return new_value, error

def improvePolicy(value, prob):
    new_prob = np.copy(prob)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] != 0:
                next_state = np.empty(4)
                next_state.fill(-1000)
                if i > 0:
                    next_state[0] = value[i-1][j]
                if i < prob.shape[0]-1:
                    next_state[1] = value[i+1][j]
                if j > 0:
                    next_state[2] = value[i][j-1]
                if j < prob.shape[1]-1:
                    next_state[3] = value[i][j+1]
                new_prob[i][j].fill(0)
                total = 0
                for k in range (4):
                    if next_state[k] == max(next_state):
                        new_prob[i][j][k] = 1.0
                        total = total + 1
                if total == 0:
                    total = 1
                new_prob[i][j] = np.divide(new_prob[i][j], total)
    #print(new_prob)
    return new_prob

def printDir(prob):
    for i in range(prob.shape[0]):
        dir = ""
        for j in range(prob.shape[1]):
            if prob[i][j][0] > 0.1 :
                dir = dir + "^"
            else:
                dir = dir + " "
            if prob[i][j][1] > 0.1 :
                dir = dir + "v"
            else:
                dir = dir + " "
            if prob[i][j][2] > 0.1 :
                dir = dir + "<"
            else:
                dir = dir + " "
            if prob[i][j][3] > 0.1 :
                dir = dir + ">"
            else:
                dir = dir + " "
            dir = dir + " | "
        print(dir+ " ")
        print("-----------------------------------------")

for i in range (4) :
    print("Testcase #" + str(i+1))
    map = np.copy(environment.map[i])
    print(map)
    prob = initProb(map)

    it = 1
    value, error = evalPolicy(map, map, prob)
    while (error > 0.000001) | (error < - 0.000001):
        value, error = evalPolicy(map, value, prob)
        prob = improvePolicy(value, prob)
        it = it + 1
        #print("Iteration #" + str(it))
        #print(value)
        #printDir(prob)

    print("No of iteration: " + str(it))
    print(value)
    printDir(prob)
    print("\n")
  