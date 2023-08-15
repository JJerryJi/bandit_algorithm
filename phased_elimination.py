import numpy as np
import math
import matplotlib.pyplot as plt
n_arms = 100
n_theta = 100
n_features = 5
num_action_list = 20

# np.random.seed(42)
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
# theta_hat = np.zeros(n_features)


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

# X: action set A
X = np.array([g_theta(a) for a in theta_capital])
print("Action SET X: ")
print(X)
X_ = np.array([[v, g_theta(v)] for v in theta_capital])
# print("Action SET X_: ")
# print(X_)

# update v_pi
def update_v_pi(pi, A):
    v_pi = 0.001*np.eye(n_features)
    for i in range(len(A)):
        v_pi += np.outer(A[i], A[i]) * pi[i] 
    return v_pi

# input: A: action set
# output: return Pi 
def g_optimal_design(A):
    pi = np.full(len(A), 1/len(A))
    v_pi = update_v_pi(pi, A) 
    while np.max([np.dot(np.dot(a.T, np.linalg.inv(v_pi)), a)for a in A]) > (1 + .01) * n_features:
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
            self.T_l = np.array([math.ceil(2 * n_features * pi[i] / (self.epsilon_l*self.epsilon_l) * math.log(n_theta * self.l*(self.l+1) * T)) for i in range (len(self.A))])
            self.V_l = self.calculate_V_l()
            self.product_At_r_t = np.zeros(n_features)
            self.cur_action_idx = 0

            # update l, epsilon_l
            self.l += 1
            self.epsilon_l = 2 ** (-self.l)
        else:
            self.product_At_r_t += self.A[self.cur_action_idx] * r_t 

T = 50000
bandit = phase_elimination(X, T)
reg1_t = []
for i in range(T):
    x_t = bandit.run()
    # print('x_t', x_t)
    if x_t is None:
        i -= 1
        bandit.feed_reward()
        continue 
    theta_t = g_theta_inverse(x_t, X_)

    A_t = action_sets[np.random.randint(0, num_action_list)]
    idx = np.argmax([np.dot(np.transpose(a), theta_t) for a in A_t])
    a_t = A_t[idx]

    # reward:
    noise = np.random.normal(0.0, 1.0)
    r_t = np.dot(np.transpose(a_t), theta_star) + noise
    bandit.feed_reward(r_t)

    # optimal reward:
    optimal_reward = max([np.dot(np.transpose(a), theta_star) for a in A_t])
    reg += optimal_reward - np.dot(np.transpose(a_t), theta_star)
    reg1_t.append(reg/math.sqrt(T))


# Generate X-axis values (time) based on the length of avg_reg1
time = list(range(len(reg1_t)))

# Plotting avg_reg1
plt.plot(time, reg1_t, label='Reduction Regret', color='blue')

# Adding labels, legend, and title
plt.xlabel('Time')
plt.ylabel('Regret/sqrt T')
plt.title('Average Regret over Time')
plt.legend()
plt.grid(True)

plt.savefig(f'PE{T}_f-{n_features}_arms-{n_arms}_nal-{num_action_list}-_theta{n_theta}.png')

plt.show()
print('reg', reg/(math.sqrt(T)))

        


    