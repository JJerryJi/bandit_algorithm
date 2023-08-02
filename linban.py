import numpy as np
import math
from tqdm import tqdm  # Import tqdm for the progress bar

n_arms = 10
n_theta = 3
n_features = 2
num_action_list = 500
np.random.seed(42)



reg = 0
theta_capital = np.random.uniform(-1, 1, (n_theta, n_features))
theta_star = theta_capital[np.random.randint(n_theta)]
action_sets = np.random.uniform(-1, 1, (num_action_list, n_arms, n_features))

# Normalize the action_sets array in place
for i in range(num_action_list):
    norm = np.linalg.norm(action_sets[i])
    action_sets[i] /= norm

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

def g_theta_inverse(x_t, X_):
    for v, g in X_:
        if np.array_equal(g, x_t):
            return v
    return None

X = np.array([g_theta(a) for a in theta_capital])
# print(X)
# print()
X_ = np.array([[v, g_theta(v)] for v in theta_capital])
print(X_)


class linBand():
    def __init__(self, action_set, T):
        self.n_features =  action_set.shape[1]
        self.T = T
        self.v = 0.001*np.eye(self.n_features)
        self.v_inv = np.linalg.inv(self.v)
        self.theta_hat = np.zeros(n_features)
        # b is now shape n_features * 1
        self.b = np.zeros(self.n_features)
        self.X =  action_set

    def UCB(self, x):
        x_transpose = np.transpose(x)
        product = np.dot(x_transpose, self.theta_hat)
        sqrt_term = np.sqrt(np.dot(np.dot(x_transpose, self.v_inv), x))
        ucb = product + 4 * sqrt_term * math.log(self.T) 
        return ucb
    
    def run(self,action_set=None):
        if action_set is not None:
            self.X = action_set[:]
        idx = np.argmax([self.UCB(x) for x in self.X])
        #x_t: the action chosen based on history
        self.x_t = self.X[idx]
        return self.x_t
    
    def feed_reward(self, r_t):
        self.v += self.x_t * self.x_t.reshape(-1,1)
        self.v_inv = np.linalg.inv(self.v)
        self.b += self.x_t * r_t  
        self.theta_hat = np.linalg.lstsq(self.v, self.b, rcond=None)[0]
        
    # def calculate_regret(self, theta_star):
        


T = 100000
# Initialize tqdm with the total number of iterations (T-1)
pbar = tqdm(total=T - 1, desc="Training Progress", unit="iteration")

Bandit = linBand(X,T)

print()

# for t in range(1, T):
#     A_t = action_sets[np.random.randint(0, num_action_list)]
    
#     # print(A_t)
#     # best action is selected
#     x_t = Bandit.run()

#     theta_t = g_theta_inverse(x_t, X_)
#     # print(theta_t)

#     # a_t
#     idx = np.argmax(np.dot(np.transpose(a), theta_t) for a in A_t)
#     a_t = A_t[idx]
#     # print(a_t)
#     #reward is generate hered: 
#     noise = np.random.normal(0.0, 1.0)
#     r_t = np.dot(np.transpose(a_t), theta_star) + noise

#     # print(r_t)
#     Bandit.feed_reward(r_t)

#     # optimal reward
#     optimal_reward = max([np.dot(np.transpose(a), theta_star) for a in A_t])
#     # print(f"best reward: {optimal_reward}")
#     reg += optimal_reward - np.dot(np.transpose(a_t), theta_star)

#     pbar.update(1)

# print(f'the reg is {reg/(math.sqrt(T) * math.log(T))}')



print()

Bandit2 = linBand(X,T)
reg = 0
for t in range(1, T):
    A_t = action_sets[np.random.randint(0, num_action_list)]
    
    # print(A_t)
    # best action is selected
    a_t = Bandit2.run(A_t)
    # print(a_t)
    #reward is generate hered: 
    noise = np.random.normal(0.0, 1.0)
    r_t = np.dot(np.transpose(a_t), theta_star) + noise
    # print(r_t)
    Bandit2.feed_reward(r_t)

    # optimal reward
    optimal_reward = max([np.dot(np.transpose(a), theta_star) for a in A_t])
    # print(f"best reward: {optimal_reward}")
    reg += optimal_reward - np.dot(np.transpose(a_t), theta_star)
    # print(reg)
    pbar.update(1)

    

print(f'the reg is {reg/(math.sqrt(T) * math.log(T))}')