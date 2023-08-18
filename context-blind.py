import numpy as np
import math
import matplotlib.pyplot as plt

# initialization of settings:
n_arms = 10
n_theta = 10
n_features = 2
num_action_list = 4

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

def update_v_pi(pi, A):
    v_pi = 0.001*np.eye(n_features)
    for i in range(len(A)):
        v_pi += np.outer(A[i], A[i]) * pi[i]
    return v_pi
def g_optimal_design(A):
    # A contains only one arm
    pi = np.full(len(A), 1/len(A))
    v_pi = update_v_pi(pi, A)
    while np.max([np.dot(np.dot(a.T, np.linalg.inv(v_pi)), a) for a in A]) > (1 + .01) * n_features:
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


def ExplorationPolicy(Z, T):
    # initalization:
    k = 1 / (T * T)
    U = np.eye(n_features) * k
    n = 1 
    T_n = []
    W_n = U 
    m = len(Z)
    # define L here: 
    L = 1/(200 * math.log(T * T * n_features))
    # list to store T_idx and W_idx:
    list_W = []
    list_T = []

    for i in range(m):
        T_n.append(i)
        # print('number of left arms', len(Z[i]))
        # choose z_i in Z[i] that maximize: 
        W_n_inv = np.linalg.inv(W_n)
        max_idx = np.argmax([np.dot(np.dot(z.T, W_n_inv), z) for z in Z[i]])
        z_i = Z[i][max_idx]

        value = math.sqrt(L/(np.dot(np.dot(z_i.T, W_n_inv), z_i))) 
        # print('value', value)   
        z_i_tilde = min(1, value) * z_i 
        # print('outer', np.outer(z_i_tilde, z_i_tilde))
        U = U + np.outer(z_i_tilde, z_i_tilde)
        # print('l', L)
        # print('2*W_N Det', 2 * np.linalg.det(W_n))
        # print('DET U', np.linalg.det(U) )
        if np.linalg.det(U) > 2 * np.linalg.det(W_n):
            list_T.append(len(T_n)/m)
            list_W.append(W_n)
            # update & reset 
            n += 1
            T_n = []
            W_n = U 

    # update the final T and W 
    list_T.append(len(T_n)/m)
    list_W.append(W_n)
    return list_W, list_T
     

def find_T_tleda(T, M):
    d_telda = n_features * math.log(T * T * n_features * n_arms) * math.log(T * T/Lambda)
    # M: batch setting 
    
    h = min(n_arms, n_features)
    T_tleda = []  

    if d_telda <= T <= d_telda * h ** (2- 2**(-M+2)):
        r = math.ceil(T ** (1/(2- 2**(-M+2))) * d_telda ** ((1 - 2**(-M+2))/(2 - 2**(-M+2))))
        assert(r >= d_telda)
        # initialize T1, T2
        T_tleda.append(r)
        T_tleda.append(r)
        for t in range(2, M):
            T_tleda.append(math.ceil(r * math.sqrt(T_tleda[t-1])/math.sqrt(d_telda)))
    elif T > d_telda * h ** (2- 2**(-M+2)): 
        r = math.ceil(T ** (1/(2- 2**(-M+2))) * d_telda ** ((1 - 2**(-M+2))/(2 - 2**(-M+2))) * h ** ((2**(-M)-1)/(2-2**(-M+1))))
        assert(r >= d_telda)
        T_tleda.append(r)
        T_tleda.append(r)
        for t in range(2, M):
            T_tleda.append(math.ceil(r * math.sqrt(T_tleda[t-1])/math.sqrt(d_telda * h)))

    return T_tleda

def natural_elimination(A, delta, theta_hat):
    delta_inv = np.linalg.inv(delta)
    res = []
    alpha = math.sqrt(50 * math.log(n_arms * T * T * n_features))
    for a in A: 
        for y in A:
            if np.array_equal(a, y):
                continue
            first_part = np.dot(a.T, theta_hat) + alpha * math.sqrt(np.dot(np.dot(a.T,delta_inv), a))
            second_part =  np.dot(y.T, theta_hat) - alpha * np.dot(np.dot(y.T,delta_inv), y)
            if first_part >= second_part:
                res.append(a)
                break 
    res_array = np.array(res)  
    return res_array

# initalize: 
T = 10000
Lambda = 10/T 
delta = Lambda * np.eye(n_features) 
M = math.ceil(math.log(math.log(T)))
T_tleda = find_T_tleda(T, M)
print(T_tleda)
list_A_t = []
list_y_t = []
list_r_t = []
list_delta = [None] * (M+1)
list_delta[0] = delta

list_theta_hat = [None] * (M+1) 
list_theta_hat[0] = theta_hat

for t in range(T_tleda[0]):
    #randomly chosen an action set
    A_t = action_sets[np.random.randint(0,num_action_list)]
    list_A_t.append(A_t)
    #we need to pass A_t as a copy, as A_t is changed inside
    pi = g_optimal_design(A_t)
    # randomly select a random arm from A_t according to g_optimal design 
    selected_idx = np.random.choice(len(A_t), p=pi)
    a_t = A_t[selected_idx]

    # calcualte reward
    noise = np.random.normal(0.0, 1.0)
    r_t = np.dot(a_t, theta_star) + noise 

    # save r_t and a_t
    list_y_t.append(a_t)
    list_r_t.append(r_t)

list_delta[1] = delta + sum([np.outer(list_y_t[i], list_y_t[i]) for i in range(math.ceil(T_tleda[0]/2))])
delta1_inverse = np.linalg.inv(list_delta[1])
list_theta_hat[1] = np.dot(delta1_inverse, sum([list_r_t[i] * list_y_t[i] for i in range(math.ceil(T_tleda[0]/2))]))

Z = []
for i in range(math.ceil(T_tleda[0]/2), T_tleda[0]):
    Z_i = natural_elimination(list_A_t[i], list_delta[1], theta_hat)
    Z.append(Z_i)


# initialzing to be pi_2
pi_k = ExplorationPolicy(Z, T)
for k in range(1, M):
    for t in range(T_tleda[k-1]+1, T_tleda[k-1]+T_tleda[k]+1):
        # observe X_t
        A_t = action_sets[np.random.randint(0,num_action_list)]
        A_t_k = []
        for i in range(1, k+1):
            # first iteration
            if i == 1:
                A_t_k = natural_elimination(A_t, list_delta[i], list_theta_hat[i])
                print('first iteration', A_t_k)
                continue 
            for new_arr in natural_elimination(A_t, list_delta[i], list_theta_hat[i]):
                A_t_k = [arr for arr in A_t_k if not np.array_equal(arr, new_arr)] 
                # # if only one arm left: 
                # if len(A_t_k) == 1:
                #     break
                print('each', i, A_t_k)

        print('res each time',  A_t_k)


        list_A_t.append(A_t_k)

        # play the arm with the feature vector y_t given pi_k(A_t_k)
        list_w, list_prob = pi_k

        selected_idx = np.random.choice(len(list_w), p=list_prob)
        W_j_inv = np.linalg.inv(list_w[selected_idx])
        selected_idx = np.argmax([np.dot(np.dot(x.T, W_j_inv), x) for x in A_t_k])

        #featured vector
        y_t = A_t_k[selected_idx]
        list_y_t.append(y_t)

        #receive a reward 
        r_t = np.dot(y_t, theta_star)
        list_r_t.append(r_t)
    
    # print('len(list_r_t)', len(list_r_t))
    # print('T_tleda[k-1]', T_tleda[k-1])
    # print('T_tleda[k-1] + T_tleda[k]//2', T_tleda[k-1]  + T_tleda[k]//2)

    delta_k = delta + sum([np.outer(list_y_t[i], list_y_t[i]) for i in range(T_tleda[k-1]+1, T_tleda[k-1] + T_tleda[k]//2)])
    list_delta[k+1] = delta_k

    delta_k_inverse = np.linalg.inv(delta_k)
    list_theta_hat[k+1] =  np.dot(delta_k_inverse, sum([list_r_t[i] * list_y_t[i] for i in range(T_tleda[k-1]+1, T_tleda[k-1] + T_tleda[k]//2)]))

    X_k_1 = []
    for t in range(T_tleda[k-1]+1, T_tleda[k-1] + T_tleda[k]//2):
        X_k_1_t = []
        for i in range(1, k+1):
            # first iteration
            if i == 1:
                X_k_1_t = natural_elimination(list_A_t[t], list_delta[i], list_theta_hat[i])
                continue 
            for new_arr in natural_elimination(list_A_t[t], list_delta[i], list_theta_hat[i]):
                X_k_1_t = [arr for arr in X_k_1_t if not np.array_equal(arr, new_arr)] 
        X_k_1.append(X_k_1_t)
    
    print(X_k_1)

    pi_k = ExplorationPolicy(X_k_1, T)

print(theta_star)
print(list_theta_hat[-1])