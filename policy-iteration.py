import numpy as np
import math
import environment

def initProb(reward): # Initialize policy with all equal probability
    prob = np.empty((reward.shape[0], reward.shape[1], 4))
    #print(reward)
    for i in range(reward.shape[0]):
        for j in range(reward.shape[1]):
            if reward[i][j] == 0:
                prob[i][j].fill(0.0)
            else:
                prob[i][j].fill(1.0)
                if i == 0:              # first row
                    prob[i][j][0] = 0.0
                if i == reward.shape[0]-1: # last row
                    prob[i][j][1] = 0.0
                if j == 0:              # first col
                    prob[i][j][2] = 0.0
                if j == reward.shape[1]-1: # last col
                    prob[i][j][3] = 0.0
                np.divide(prob[i][j], np.sum(prob[i][j]))

    return prob

def evalPolicy(reward, value, prob):
    new_value = np.copy(value)
    error = 0
    for i in range(reward.shape[0]):
        for j in range(reward.shape[1]):
            new_value[i][j] = reward[i][j]
            if i > 0:
                new_value[i][j] += prob[i][j][0] * value[i-1][j]
            if i < prob.shape[0]-1:
                new_value[i][j] += prob[i][j][1] * value[i+1][j]
            if j > 0:
                new_value[i][j] += prob[i][j][2] * value[i][j-1]
            if j < prob.shape[1]-1:
                new_value[i][j] += prob[i][j][3] * value[i][j+1]
            error += value[i][j] - new_value[i][j]
            #print(value[i][j], new_value[i][j])
    #print(new_value)
    return new_value, error

def improvePolicy(value, prob):
    new_prob = np.copy(prob)
    for i in range(reward.shape[0]):
        for j in range(reward.shape[1]):
            if reward[i][j] != 0:
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
                for k in range (4):
                    if next_state[k] == max(next_state):
                        new_prob[i][j][k] = 1.0
                new_prob[i][j] = np.divide(new_prob[i][j], np.sum(new_prob[i][j]))
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

# Main #

for i in range (4) :
    print("Testcase #" + str(i+1))
    reward = np.copy(environment.map[i])
    print(reward)
    prob = initProb(reward)

    it = 1
    value, error = evalPolicy(reward, reward, prob)
    while (error > 0.000001) | (error < - 0.000001):
        value, error = evalPolicy(reward, value, prob)
        prob = improvePolicy(value, prob)
        it = it + 1
        #print("Iteration #" + str(it))
        #print(value)
        #printDir(prob)

    print("No of iteration: " + str(it))
    print(value)
    printDir(prob)
    print("\n")