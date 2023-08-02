import numpy as np
import math 
num_action_list = 10
n_theta = 3
n_arms = 3
n_features = 2

np.random.seed(42)
action_sets = np.random.uniform(-1, 1, (num_action_list, n_arms, n_features))
theta_capital = np.random.uniform(-1, 1, (n_theta, n_features))
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
    v_pi = np.zeros((n_features, n_features))
    for i in range(len(A)):
        v_pi += A[i] * A[i].reshape(-1, 1) * pi[i] 
    return v_pi



# input: A: action set
# output: return Pi 
def g_optimal_design(A):
    # A contains only one arm 
    pi = np.full(len(A), 1/len(A))
    v_pi = update_v_pi(pi, A) 
    while np.max([np.dot(a.T, np.linalg.inv(v_pi), a)for a in A]) > (1 + .01) * n_features:
        # find ak
        idx = np.argmax([np.dot(np.dot(a.T, np.linalg.inv(v_pi)), a)for a in A])
        a_k = A[idx]
        # print('here is a_k')
        # print(a_k)

        # find gamma_k 
        numerator= (1/n_features) * np.dot(np.dot(a_k.T, np.linalg.inv(v_pi)), a_k) - 1
        denominator = np.dot(np.dot(a_k.T, np.linalg.inv(v_pi)), a_k) - 1
        gamma_k = numerator / denominator

        #update pi
        pi *= (1-gamma_k) 
        pi[idx] += gamma_k
        # print('here is pi:')
        # print(pi)

        #update v_pi:
        v_pi = update_v_pi(pi, A) 
    return pi

l = 1 
A1 = X
T = 3000
t = 1
reg = 0
while t < T:
    # step 1
    pi = g_optimal_design(A1)
    # print("here is pi")
    # print(pi)
    epsilon_l = 2 ** (-l)

    # step 2
    T_l =  np.array([math.ceil(2 * n_features * pi[i] / (epsilon_l*epsilon_l) * math.log(n_theta * l*(l+1) * t)) for i in range (len(A1))])
    # print(T_l)

    # step 3 & step 4
    for i in range (len(A1)):
        # calculate V_l:
        V_l = np.zeros((n_features, n_features))
        for j in range(len(A1)):
            V_l += np.outer(A1[j], A1[j]) * T_l[j]
        
        V_l_inverse = np.linalg.inv(V_l)

        temp = np.zeros(n_features)
        for _ in range(T_l[i]):
            noise = np.random.normal(0.0, 1.0)
            r_t = np.dot(np.transpose(A1[i]), theta_star) + noise
            t+=1
            optimal_reward = max([np.dot(np.transpose(a), theta_star) for a in A1])
            # print(f"best reward: {optimal_reward}")
            reg += optimal_reward - np.dot(np.transpose(A1[i]), theta_star)
            # print(f'the reg is {reg/(math.sqrt(T) * math.log(T))}')
            temp+= A1[i] * r_t 

        # print(V_l_inverse)
        # print(temp)
        theta_hat =  np.dot(V_l_inverse, temp.reshape(-1, 1)).reshape(1, -1)
        # print(r_t)

        # step 5
        res = []
        for a in A1:
            if any(np.array_equal(row, a) for row in res): continue 

            if max([np.dot(theta_hat, b-a)for b in A1]) <= 2 * epsilon_l:
                res.append(a)
        
    A1 = np.array(res)

    if len(A1) == 1:
        print(t)
        break
    print("A1 is updated")
    print(A1)
    # step 6 
    l += 1


 
while t<T:
    optimal_reward = max([np.dot(np.transpose(a), theta_star) for a in A1])
    # print(f"best reward: {optimal_reward}")
    reg += optimal_reward - np.dot(np.transpose(A1[0]), theta_star)
    print(f'the reg is {reg/(math.sqrt(T) * math.log(T))}')
    t+=1
