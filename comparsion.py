import numpy as np
import math
import matplotlib.pyplot as plt

# initialization of settings:
n_arms = 500
n_theta = 50
n_features = 5
num_action_list = 200


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
    while np.max([np.dot(a.T, np.linalg.inv(v_pi), a)for a in A]) > (1 + .01) * n_features:
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
    def __init__(self, action_set, T, l=1, epsilon_l=2**(-1)) -> None:
        self.A = action_set
        # keep track which action to play
        self.cur_action_idx = 0
        # caculate T_l[], V_l, product_At_r_t
        pi = g_optimal_design(self.A.copy())
        self.T_l = np.array([math.ceil(2 * n_features * pi[i] / (epsilon_l*epsilon_l)
                            * math.log(n_theta * l*(l+1) * T)) for i in range(len(self.A))])
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
            self.cur_action_idx += 1

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
            self.theta_hat = np.dot(
                V_l_inverse, self.product_At_r_t.reshape(-1, 1)).reshape(1, -1)
            # Eliminate arms:
            res = []
            # print(self.A)
            for a in self.A:
                if any(np.array_equal(row, a) for row in res):
                    continue

                if max([np.dot(self.theta_hat, b-a)for b in self.A]) <= 2 * self.epsilon_l:
                    res.append(a)
            self.A = np.array(res)
            # print(self.A)
            # reset T_l, V_l, product-At*rt, cur_action_idx
            pi = g_optimal_design(self.A.copy())
            self.T_l = np.array([math.ceil(2 * n_features * pi[i] / (self.epsilon_l*self.epsilon_l)
                                * math.log(n_theta * self.l*(self.l+1) * self.T)) for i in range(len(self.A))])
            self.V_l = self.calculate_V_l()
            self.product_At_r_t = np.zeros(n_features)
            self.cur_action_idx = 0

            # update l, epsilon_l
            self.l += 1
            self.epsilon_l = 2 ** (-self.l)
        else:
            self.product_At_r_t += self.A[self.cur_action_idx] * r_t


def g_theta(theta):
    '''
    g_theta(theta): Reduction Algorithm 
        @param: theta 
        output: to generate fix action set
    '''
    # initialize g_theta
    g_theta = np.zeros(n_features)
    # iterate through each arm in action_sets
    for i in range(num_action_list):
        # find the best arm
        best_action = np.argmax([np.dot(a, theta) for a in action_sets[i]])
        # calculate expectation
        g_theta += action_sets[i][best_action] * (1 / num_action_list)
    return g_theta


def g_theta_inverse(x_t, X_):
    '''
    g_theta_inverse(x_t, X_): 
        @param: x_t, X_
            x_t: g(theta) 
            X_:  look-up table
        output: the theta that's associated with g(theta)
    '''
    for v, g in X_:
        if np.array_equal(g, x_t):
            return v
    return None


# reg1: the regret of reduction algorithm
# reg2: the regret of original algorithm
reg1 = 0
reg2 = 0
T = 50000

iterations = 10

list_reg1 = []
list_reg2 = []

for i in range(iterations):
    # to show progress:
    print('Progress:', (i+1)/iterations)

    reg = 0
    theta_capital = np.random.normal(0, 1, (n_theta, n_features))
    action_sets = np.random.normal(0, 1,
                                   (num_action_list, n_arms, n_features))
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

    # fixed action set
    X = np.array([g_theta(a) for a in theta_capital])
    # look-up table between each theta and the generated action
    X_ = np.array([[v, g_theta(v)] for v in theta_capital])

    # Bandit Instance for Reduction Case:
    Bandit = phase_elimination(X, T)

    # reg1_t: contains [1*T] regret for reduction case
    reg1_t = []
    for t in range(T):
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

        reg1_t.append(reg)

    list_reg1.append(reg1_t)
    reg1 += reg/iterations

    # Bandit Instance for Original Case:
    Bandit2 = linBand(X, T)
    # reset regret
    reg = 0
    # reg2_t: contains [1*T] regret for original case
    reg2_t = []
    for t in range(T):
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
        reg2_t.append(reg)

    list_reg2.append(reg2_t)
    reg2 += reg/iterations


def calculate_avg_regret_list(ls):
    """
    Calculate the average regret for each column (element-wise) in a list of lists.

    Args:
        ls (list of lists): List containing sublists representing regret values for each time step. 

    Returns:
        list: A list containing the average regret over sqrt(T) for each column.
    """
    if not ls:
        return []

    num_cols = len(ls[0])
    avg_regrets = [0] * num_cols

    for row in ls:
        for i, value in enumerate(row):
            avg_regrets[i] += value

    avg_regrets = [avg / (len(ls) * math.sqrt(T)) for avg in avg_regrets]
    return avg_regrets


avg_reg1 = calculate_avg_regret_list(list_reg1)
avg_reg2 = calculate_avg_regret_list(list_reg2)

# sanity check
assert(len(avg_reg1) == T)
assert(len(avg_reg2) == T)

print(f'the reduction reg is {reg1/(math.sqrt(T))}',
      f'the original reg is {reg2/(math.sqrt(T))}')


# Generate X-axis values (time) based on the length of avg_reg1
time = list(range(len(avg_reg1)))

# Plotting avg_reg1
plt.plot(time, avg_reg1, label='Reduction Regret', color='blue')

# Plotting avg_reg2
plt.plot(time, avg_reg2, label='Original Regret', color='orange')

# Adding labels, legend, and title
plt.xlabel('Time')
plt.ylabel('Regret/sqrt T')
plt.title('Average Regret over Time')
plt.legend()

# Add a grid
plt.grid(True)  # This line adds a grid to the plot

# Save the plot to a file (change the filename and format as needed)
plt.savefig(
    f'result/Comparsion{T}_f-{n_features}_arms-{n_arms}_nal-{num_action_list}-I{iterations}_theta{n_theta}.svg',  format="svg")

# Display the plot
plt.show()
