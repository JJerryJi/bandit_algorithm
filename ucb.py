import numpy as np 
import math 

n_features = 5 

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

    def UCB(self, x):
        '''
        UCB():
            @param:
            x: an action
            output: return the reward of the input action by Upper-confidence-bound method
        '''
        x_transpose = np.transpose(x)
        product = np.dot(x_transpose, self.theta_hat)
        sqrt_term = np.sqrt(np.dot(np.dot(x_transpose, self.v_inv), x))
        ucb = product + 4 * sqrt_term * math.log(self.T)
        return ucb

    def run(self, action_set=None):
        '''
        run():
            @param:
                action_set(optional): only the original algorithm  will provide action_set

            output: return the selected action by calling UCB() method
        '''
        if action_set is not None:
            self.X = action_set
        idx = np.argmax([self.UCB(x) for x in self.X])
        # x_t: the action chosen based on history
        self.x_t = self.X[idx]
        return self.x_t

    def feed_reward(self, r_t):
        '''
        feed_reward():
            @param: 
            r_t: reward for the current action 
            output: update v, b, and theta_hat; no return  
        '''
        self.v += np.outer(self.x_t, self.x_t)
        self.v_inv = np.linalg.inv(self.v)
        self.b += self.x_t * r_t
        self.theta_hat = np.linalg.lstsq(self.v, self.b, rcond=None)[0]