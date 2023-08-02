import numpy as np
import math
from pprint import pprint

n_arms = 10
n_theta = 3
n_features = 2
num_action_list = 500

np.random.seed(42)

v = 0.001*np.eye(n_features)
v_inv = np.linalg.inv(v)
# b is now shape n_features * 1
b = np.zeros(n_features)
reg = 0
theta_capital = np.random.uniform(-1, 1, (n_theta, n_features))
theta_hat = np.zeros(n_features)
theta_star = theta_capital[np.random.randint(n_theta)]
action_sets = np.random.uniform(-1, 1, (num_action_list, n_arms, n_features))

# print(f"theta capital is: {theta_capital}")


def g_theta(theta):
    # initialize g_theta
    g_theta = np.zeros(n_features)
    # iterate through each set in action_sets
    for i in range(num_action_list):
        # find the best_action
        best_action = np.argmax([np.dot(a, theta) for a in action_sets[i]])
        # calculate expectation
        g_theta += action_sets[i][best_action] * (1 / num_action_list) 
    return g_theta



X = np.array([g_theta(a) for a in theta_capital])
print(X)
print()
def UCB(x, theta_hat, v_inv, T):
    x_transpose = np.transpose(x)
    product = np.dot(x_transpose, theta_hat)
    sqrt_term = np.sqrt(np.dot(np.dot(x_transpose, v_inv), x))
    ucb = product + 2 * sqrt_term * math.log(T) 
    return ucb

T = 100000
for t in range(1, T):
    idx = np.argmax([UCB(x, theta_hat, v_inv, T) for x in X])
    # print(f"idx is: {idx}")
    x_t = X[idx]
    # print(f"x_t is: {x_t}")
    noise = np.random.normal(0.0, 1.0)
    # print(f"theta star is: {theta_star}")
    r_t = np.dot(np.transpose(x_t), theta_star) + noise
    # print(f"r_t is: {r_t}")
    # update:
    # print(v)
    v += x_t * x_t.reshape(-1,1)
    # print(x_t * np.transpose(x_t))
    # print('here is v')
    # print(v)
    

    v_inv = np.linalg.inv(v)
    # print(f"v_inv: {v_inv}")
    b += x_t * r_t
    # print('here is b')
    # print(b)
    # theta_hat = np.dot(v_inv, b)
    theta_hat = np.linalg.lstsq(v, b, rcond=None)[0]
    # print(f"theta_hat is: {theta_hat}")
    # optimal reward
    idx = np.argmax([np.dot(np.transpose(x), theta_star) for x in X])
    # print(f"theta_star is: {theta_star}")
    optimal_reward = np.dot(np.transpose(X[idx]), theta_star)
    # print(f"The optimal reward is: {optimal_reward}")
    # print(f"The current reward is: {np.dot(np.transpose(x_t), theta_star)}")
    # calculate regret
    reg += optimal_reward - np.dot(np.transpose(x_t), theta_star)
    # print()

print(f'the reg is {reg/math.sqrt(T)}')