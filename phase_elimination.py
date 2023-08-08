import numpy as np
import math
n_arms = 500
n_theta = 3
n_features = 5
num_action_list = 100

np.random.seed(42)
action_sets = np.random.normal(0, 1, (num_action_list, n_arms, n_features))
theta_capital = np.random.normal(0, 1, (n_theta, n_features))

# Normalize the action_sets array in place
for i in range(num_action_list):
    for arm in range(n_arms):
        norm = np.linalg.norm(action_sets[i][arm])
        action_sets[i][arm] /= norm

# Normalize theta_captial
for i in range(n_theta):
    norm = np.linalg.norm(theta_capital[i])
    theta_capital[i] /= norm

theta_star = theta_capital[np.random.randint(n_theta)]
theta_hat = np.zeros(n_features)


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


# X: action set A
X = np.array([g_theta(a) for a in theta_capital])
print("Action SET X: ")
print(X)
print()

# update v_pi
def update_v_pi(pi, A):
    v_pi = 0.001*np.eye(n_features)
    for i in range(len(A)):
        v_pi += np.outer(A[i], A[i]) * pi[i] 
    return v_pi

# input: A: action set
# output: return Pi 
def g_optimal_design(A):
    # A contains only one arm 
    pi = np.full(len(A), 1/len(A))
    v_pi = update_v_pi(pi, A) 
    print('v_pi', v_pi)
    while np.max([np.dot(a.T, np.linalg.inv(v_pi), a)for a in A]) > (1 + .01) * n_features:
        # find ak
        idx = np.argmax([np.dot(np.dot(a.T, np.linalg.inv(v_pi)), a)for a in A])
        a_k = A[idx]

        # find gamma_k 
        numerator= (1/n_features) * np.dot(np.dot(a_k.T, np.linalg.inv(v_pi)), a_k) - 1
        denominator = np.dot(np.dot(a_k.T, np.linalg.inv(v_pi)), a_k) - 1
        gamma_k = numerator / denominator

        #update pi
        pi *= (1-gamma_k) 
        pi[idx] += gamma_k

        #update v_pi:
        v_pi = update_v_pi(pi, A) 
    return pi

reg = 0
l = 1
epsilon_l = 2 ** (-l)
T = 1000

class phase_elimination():
    def __init__(self, action_set) -> None:
        self.A = action_set
        # keep track which action to play 
        self.cur_action_idx = 0
        # caculate T_l[]
        pi = g_optimal_design(self.A)
        self.T_l = np.array([math.ceil(2 * n_features * pi[i] / (epsilon_l*epsilon_l) * math.log(n_theta * l*(l+1) * T)) for i in range (len(self.A))])
        # initalization
        self.V_l = self.calculate_V_l()
        self.product_At_r_t = np.zeros(n_features)
    
    # calcualte the V_l
    def calculate_V_l(self):
        temp = np.zeros((n_features, n_features))
        for i in range(len(self.A)):
            temp += self.T_l[i] * np.outer(self.A[i], self.A[i])
        return temp

    def run(self):
        global l, epsilon_l, reg
        # only one arm left
        if len(self.A) == 1:
            optimal_reward = max([np.dot(np.transpose(a), theta_star) for a in self.A])
            # print(f"best reward: {optimal_reward}")
            reg += optimal_reward - np.dot(np.transpose(self.A[0]), theta_star)
            print(f'the reg is {reg/(math.sqrt(T) * math.log(T))}')
            return 
        
        if np.all(self.T_l == 0):
            # calculate theta_hat: 
            V_l_inverse = np.linalg.inv(self.V_l)
            theta_hat =  np.dot(V_l_inverse, self.product_At_r_t.reshape(-1, 1)).reshape(1, -1)

            # Eliminate arms: 
            res = []
            for a in self.A:
                if any(np.array_equal(row, a) for row in res): continue 

                if max([np.dot(theta_hat, b-a)for b in self.A]) <= 2 * epsilon_l:
                    res.append(a)
            self.A = res

            # reset T_l, V_l, temp, cur_action_idx
            pi= g_optimal_design(self.A)
            self.T_l = np.array([math.ceil(2 * n_features * pi[i] / (epsilon_l*epsilon_l) * math.log(n_theta * l*(l+1) * T)) for i in range (len(self.A))])
            self.V_l = self.calculate_V_l()
            self.product_At_r_t = np.zeros(n_features)
            self.cur_action_idx = 0

            # update l, epsilon_l
            l+=1
            epsilon_l = 2 ** (-l)


        if self.T_l[self.cur_action_idx] == 0:
            self.cur_action_idx+=1
        

        noise = np.random.normal(0.0, 1.0)
        r_t = np.dot(np.transpose(self.A[self.cur_action_idx]), theta_star) + noise

        optimal_reward = max([np.dot(np.transpose(a), theta_star) for a in self.A])
        # print(f"best reward: {optimal_reward}")
        reg += optimal_reward - np.dot(np.transpose(self.A[self.cur_action_idx]), theta_star)

        print(f'the reg is {reg/(math.sqrt(T) * math.log(T))}')

        self.product_At_r_t += self.A[self.cur_action_idx] * r_t 

        self.T_l[self.cur_action_idx] -= 1

bandit = phase_elimination(X)
for i in range(T):
    bandit.run()

        

    