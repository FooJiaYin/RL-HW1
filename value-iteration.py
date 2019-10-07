import numpy as np
import math
import environment

def evalValue(reward, value):
    new_value = np.copy(reward)
    error = 0
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            if reward[i][j] != 0:
                next_state = np.empty(4)
                next_state.fill(-1000)
                if i > 0:
                    next_state[0] = value[i-1][j]
                if i < value.shape[0]-1:
                    next_state[1] = value[i+1][j]
                if j > 0:
                    next_state[2] = value[i][j-1]
                if j < value.shape[1]-1:
                    next_state[3] = value[i][j+1]
                new_value[i][j] = reward[i][j] + max(next_state)
                error += value[i][j] - new_value[i][j]
    return new_value, error

# Main #

for i in range (4) :
    print("Testcase #" + str(i+1))
    reward = np.copy(environment.map[i])
    print(reward)

    it = 1
    value, error = evalValue(reward, reward)
    while (error > 0.000001) | (error < - 0.000001):
        value, error = evalValue(reward, value)
        it = it + 1
        #print("Iteration #" + str(it))
        #print(value)

    print("No of iteration: " + str(it))
    print(value)
    print("\n")
