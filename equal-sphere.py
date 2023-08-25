import numpy as np
import math
from angle import cartesian_to_spherical, spherical_to_cartesian
import collections

# np.random.seed(42)
n_arm = 2
n_features = 3
n_sector = 9

def find_partition(num_sector):
    for i in range(1, num_sector):
        number_of_partition = i ** (n_features - 1)
        if 0.75 * num_sector <= number_of_partition < num_sector * 1.5:
            return i, number_of_partition
    return None

step_size, num_partitions = find_partition(n_sector)
print('num_partition', num_partitions)
print('step_size', step_size)
phi_general = np.linspace(0, np.pi , num = (step_size + 1))
phi_2PI = np.linspace(0, 2 * np.pi, num = (step_size + 1))
print(phi_general)
print(phi_2PI)

def index_phi(phi_value):
    for i in range(len(phi_general)-1):
        if phi_general[i] <= phi_value < phi_general[i+1]:
            return i
    # return len(phi_general) - 2

def index_theta(theta_value):
    for i in range(len(phi_2PI)-1):
        if phi_2PI[i] <= theta_value < phi_2PI[i+1]:
            return i
    # return len(phi_2PI) - 2

action_sets = [[] for _ in range(num_partitions)]

count_table = [0] * (n_features - 1)
current_row = 0

for i in range(num_partitions):

    # repeatedly generating actions: 
    while len(action_sets[i]) < n_arm: 
        action = np.random.normal(0,1, (n_features))
        action = action / np.linalg.norm(action)
        _, angles = cartesian_to_spherical(action)
        list_phi, theta_value = angles[:-1], angles[-1]

        # count current partition
        count  = 0
        for j in range(len(list_phi)):
            phi = list_phi[j]
            if index_phi(phi) != count_table[j]:
                break 
            count += 1
        if index_theta(theta_value) == count_table[-1]:
            count += 1 
        
        # current action is valid in range
        if count == n_features - 1:
            action_sets[i].append(action)

        
    
    # update each row 
    # assign a order of priority: larger idx has a larger priority 
    add_idx = None

    for k in range(len(count_table)):
        # start from the beginning, find the first place we can add 1 --> if we success, then we are done for this iteration
        if count_table[k] + 1 < step_size:
            count_table[k] += 1
            break
        # if we cannot, then add_idx = k + 1 -- > (buz count_table[:k+1] will be cleaned up as count_table[k] has been filled)
        else:
            add_idx = k+1
    
    # if need to clean up, then we will reset every vector before add_idx to be zero
    if add_idx:
        for j in range(0, add_idx):
            count_table[j] = 0
    

# check 
print(len(action_sets))
for action in action_sets:
    res = []
    for a in action:
        res.append(cartesian_to_spherical(a)[1])
    print(res)
    print()
assert all(len(action_set) == n_arm for action_set in action_sets)
