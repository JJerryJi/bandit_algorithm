import numpy as np 
import math
 
n_features = 5
n_theta = 100

def update_v_pi(pi, A):
    v_pi = 0.001*np.eye(n_features)
    for i in range(len(A)):
        v_pi += np.outer(A[i], A[i]) * pi[i]
    return v_pi

def g_optimal_design(A):
    # A contains only one arm
    pi = np.full(len(A), 1/len(A))
    v_pi = update_v_pi(pi, A)
    while np.max([np.dot(np.dot(a.T, np.linalg.inv(v_pi)), a)for a in A]) > (1 + .01) * n_features:
        # find ak
        idx = np.argmax(
            [np.dot(np.dot(a.T, np.linalg.inv(v_pi)), a)for a in A])
        a_k = A[idx]

        # find gamma_k
        numerator = (1/n_features) * \
            np.dot(np.dot(a_k.T, np.linalg.inv(v_pi)), a_k) - 1
        denominator = np.dot(np.dot(a_k.T, np.linalg.inv(v_pi)), a_k) - 1
        gamma_k = numerator / denominator

        # update pi
        pi *= (1-gamma_k)
        pi[idx] += gamma_k

        # update v_pi:
        v_pi = update_v_pi(pi, A)
    return pi


class phase_elimination():
    def __init__(self, action_set, T, l= 1, epsilon_l=2**(-1)) -> None:
        self.A = action_set
        # keep track which action to play 
        self.cur_action_idx = 0
        # caculate T_l[], V_l, product_At_r_t
        pi = g_optimal_design(self.A)
        self.T_l = np.array([math.ceil(2 * n_features * pi[i] / (epsilon_l*epsilon_l) * math.log(n_theta * l*(l+1) * T)) for i in range (len(self.A))])
        self.V_l = self.calculate_V_l()
        self.product_At_r_t = np.zeros(n_features)
        # empirical estimate
        self.theta_hat = np.zeros(n_features)
        self.l = l
        self.epsilon_l = epsilon_l
        self.T = T

    def run(self):
        # only one arm left
        if len(self.A) == 1:
            return self.A[0]
        
        # Update Everything
        if np.all(self.T_l == 0):
            return None
        
        # Normal Case: 
        if self.T_l[self.cur_action_idx] == 0:
            self.cur_action_idx+=1
        
        self.T_l[self.cur_action_idx] -= 1

        # return the chosen action
        return self.A[self.cur_action_idx]
       
    # calcualte the V_l
    def calculate_V_l(self):
        temp = 0.01 * np.eye(n_features)
        for i in range(len(self.A)):
            temp += self.T_l[i] * np.outer(self.A[i], self.A[i])
        return temp
    
    def feed_reward(self, r_t):
        if np.all(self.T_l == 0):
            # calculate theta_hat: 
            # print(self.V_l)
            # print(len(self.V_l))
            V_l_inverse = np.linalg.inv(self.V_l)
            self.theta_hat =  np.dot(V_l_inverse, self.product_At_r_t.reshape(-1, 1)).reshape(1, -1)
            # Eliminate arms: 
            res = []
            # print(self.A)
            for a in self.A:
                if any(np.array_equal(row, a) for row in res): continue 

                if max([np.dot(self.theta_hat, b-a)for b in self.A]) <= 2 * self.epsilon_l:
                    res.append(a)
            self.A = np.array(res)
            # print(self.A)
            # reset T_l, V_l, product-At*rt, cur_action_idx
            pi= g_optimal_design(self.A)
            self.T_l = np.array([math.ceil(2 * n_features * pi[i] / (self.epsilon_l*self.epsilon_l) * math.log(n_theta * self.l*(self.l+1) * self.T)) for i in range (len(self.A))])
            self.V_l = self.calculate_V_l()
            self.product_At_r_t = np.zeros(n_features)
            self.cur_action_idx = 0

            # update l, epsilon_l
            self.l += 1
            self.epsilon_l = 2 ** (-self.l)
        else:
            self.product_At_r_t += self.A[self.cur_action_idx] * r_t 