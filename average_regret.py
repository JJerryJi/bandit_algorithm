import numpy as np
import math

# initialization of settings: 
n_arms = 5
n_theta = 5
n_features = 2
num_action_list = 500

class linBand():
    def __init__(self, action_set, T):
        self.n_features = action_set.shape[1]
        self.T = T
        # b: Matrix[n_features * n_features] to prevent singular matrix
        self.v = 0.001*np.eye(self.n_features)
        self.v_inv = np.linalg.inv(self.v)
        self.theta_hat = np.zeros(n_features)
        # b: Matrix [n_features * 1]
        self.b = np.zeros(self.n_features)
        self.X = action_set

    '''
    UCB():
        @param:
            x: an action
        output: return the reward of the input action by Upper-confidence-bound method
    '''
    def UCB(self, x):
        x_transpose = np.transpose(x)
        product = np.dot(x_transpose, self.theta_hat)
        sqrt_term = np.sqrt(np.dot(np.dot(x_transpose, self.v_inv), x))
        ucb = product + 4 * sqrt_term * math.log(self.T)
        return ucb

    '''
    run():
        @param:
            action_set(optional): only the original algorithm  will provide action_set

        output: return the selected action by calling UCB() method
    '''
    def run(self, action_set=None):
        if action_set is not None:
                self.X = action_set
        idx = np.argmax([self.UCB(x) for x in self.X])
        # x_t: the action chosen based on history
        self.x_t = self.X[idx]
        return self.x_t
        
    '''
    feed_reward():
        @param: 
            r_t: reward for the current action 

        output: update v, b, and theta_hat; no return  
    '''
    def feed_reward(self, r_t):
        self.v += np.outer(self.x_t, self.x_t)
        self.v_inv = np.linalg.inv(self.v)
        self.b += self.x_t * r_t
        self.theta_hat = np.linalg.lstsq(self.v, self.b, rcond=None)[0]


'''
g_theta(theta): Reduction Algorithm 
    @param: theta 
    output: to generate fix action set
'''
def g_theta(theta):
    # initialize g_theta
    g_theta = np.zeros(n_features)
    # iterate through each arm in action_sets
    for i in range(num_action_list):
        # find the best arm 
        best_action = np.argmax([np.dot(a, theta) for a in action_sets[i]])
        # calculate expectation
        g_theta += action_sets[i][best_action] * (1 / num_action_list)
    return g_theta

'''
g_theta_inverse(x_t, X_): 
    @param: x_t, X_
        x_t: g(theta) 
        X_:  look-up table
    output: the theta that's associated with g(theta)
'''
def g_theta_inverse(x_t, X_):
    for v, g in X_:
        if np.array_equal(g, x_t):
            return v
    return None



# reg1: the regret of reduction algorithm 
# reg2: the regret of original algorithm
reg1 = 0
reg2 = 0
T = 100000

iterations = 30

for i in range(iterations):
    # to show progress: 
    print((i+1)/iterations)

    reg = 0
    theta_capital = np.random.uniform(-1, 1, (n_theta, n_features))
    theta_star = theta_capital[np.random.randint(n_theta)]
    action_sets = np.random.uniform(-1, 1,
                                    (num_action_list, n_arms, n_features))

    # Normalize the action_sets array in place
    for i in range(num_action_list):
        norm = np.linalg.norm(action_sets[i])
        action_sets[i] /= norm


    # fixed action set
    X = np.array([g_theta(a) for a in theta_capital])
    # look-up table between each theta and the generated action   
    X_ = np.array([[v, g_theta(v)] for v in theta_capital])

    # Bandit Instance for Reduction Case: 
    Bandit = linBand(X, T)

    for t in range(1, T):
        x_t = Bandit.run()
        theta_t = g_theta_inverse(x_t, X_)

        A_t = action_sets[np.random.randint(0, num_action_list)]
        idx = np.argmax([np.dot(np.transpose(a), theta_t) for a in A_t])
        a_t = A_t[idx]

        # reward:
        noise = np.random.normal(0.0, 1.0)
        r_t = np.dot(np.transpose(a_t), theta_star) + noise
        Bandit.feed_reward(r_t)

        # optimal reward:
        optimal_reward = max(
            [np.dot(np.transpose(a), theta_star) for a in A_t])
        reg += optimal_reward - np.dot(np.transpose(a_t), theta_star)

    reg1 += reg/iterations

    # Bandit Instance for Original Case: 
    Bandit2 = linBand(X, T)
    reg = 0
    for t in range(1, T):
        # the action is selected
        A_t = action_sets[np.random.randint(0, num_action_list)]
        a_t = Bandit2.run(A_t)

        # reward:
        noise = np.random.normal(0.0, 1.0)
        r_t = np.dot(np.transpose(a_t), theta_star) + noise

        Bandit2.feed_reward(r_t)

        # optimal reward:
        optimal_reward = max(
            [np.dot(np.transpose(a), theta_star) for a in A_t])
        reg += optimal_reward - np.dot(np.transpose(a_t), theta_star)

    reg2 += reg/iterations


print(f'the reduction reg is {reg1/(math.sqrt(T))}',
      f'the original reg is {reg2/(math.sqrt(T))}')
