import numpy as np
import math
import matplotlib.pyplot as plt

# initialization of settings:
n_arms = 3
n_theta = 3
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
            print('len(T_n)/m', len(T_n)/m)
            list_T.append(len(T_n)/m)
            list_W.append(W_n)
            # update & reset 
            n += 1
            T_n = []
            W_n = U 

    # fix updating logic
    print('list_T', len(list_T))
    print(n-1)
    if len(list_T) == n-1:
        print('yes: logic activated')
        # update the final T and W 
        list_T.append(len(T_n)/m)
        list_W.append(W_n)
    # print(list_W)
    # print(list_T)
    return list_W, list_T
     

def find_T(T, M):
    d_telda = n_features * math.log(T * T * n_features * n_arms) * math.log(T * T/Lambda)
    # M: batch setting 
    h = min(n_arms, n_features)
    list_T_k = []  
    if d_telda <= T <= d_telda * h ** (2- 2**(-M+2)):        
        r = math.ceil( T ** (1/(2- 2**(-M+2))) * d_telda ** ((1 - 2**(-M+2))/(2 - 2**(-M+2))))
        assert(r >= d_telda)
        # initialize T1, T2
        list_T_k.append(r)
        list_T_k.append(r)
        for t in range(2, M):
            list_T_k.append(math.ceil(r * math.sqrt(list_T_k[t-1])/math.sqrt(d_telda)))
    elif T > d_telda * h ** (2- 2**(-M+2)): 
        r = math.ceil(T ** (1/(2 - 2**(-M+1))) * d_telda ** ((1 - 2**(-M+1))/(2 - 2**(-M+1))) * h ** ((2**(-M+1)/(2-2**(-M+1)))))
        assert(r >= d_telda)
        list_T_k.append(r)
        list_T_k.append(math.ceil((r * math.sqrt(list_T_k[0]))/math.sqrt(d_telda * h)))
        for t in range(2, M):
            list_T_k.append(math.ceil(r * math.sqrt(list_T_k[t-1])/math.sqrt(d_telda)))

    return list_T_k

def find_T_tleda(T, M):
    list_T = find_T(T, M)
    T_delta = []
    current_sum = 0
    for i in range(len(list_T)):
        current_sum += list_T[i]
        T_delta.append(min(current_sum, T))
    assert(len(T_delta) == len(list_T))
    return T_delta, list_T

def natural_elimination(A, delta, theta_hat):
    # # try to prevent empty set:
    # if len(A) == 1:
    #     return A
    delta_inv = np.linalg.inv(delta)
    res = []
    alpha = math.sqrt(50 * math.log(n_arms * T * T * n_features))
    # print('A', A)
    # print('delta', delta)
    # print('theta', theta_hat)
    for a in A: 
        first_part = np.dot(a.T, theta_hat) + alpha * math.sqrt(np.dot(np.dot(a.T,delta_inv), a))
        add = True
        for y in A:
            second_part = np.dot(y.T, theta_hat) - alpha * math.sqrt(np.dot(np.dot(y.T,delta_inv), y))
            if first_part < second_part:
                add = False
                break 
        if add:
            res.append(a)
    res_array = np.array(res)  
    return res_array

# initalize: 
T = 100000
Lambda = 10/T 
delta = Lambda * np.eye(n_features) 
M = math.ceil(math.log(math.log(T)))
print(M)
T_tleda, list_T = find_T_tleda(T, M)
print('list_T', list_T)
print('T_tleda', T_tleda)
list_A_t = []
list_y_t = []
list_r_t = []
list_delta = [None] * M
list_delta[0] = delta

list_theta_hat = [None] * M 
list_theta_hat[0] = theta_hat

reg = 0
list_reg = []

for t in range(T_tleda[0]):
    #randomly chosen an action set
    A_t = action_sets[np.random.randint(0,num_action_list)]
    list_A_t.append(A_t)
    #we need to pass A_t as a copy, as A_t is changed inside
    pi = g_optimal_design(A_t)
    # randomly select a random arm from A_t according to g_optimal design 
    pi = pi/sum(pi)
    selected_idx = np.random.choice(len(A_t), p=pi)
    a_t = A_t[selected_idx]
   
    # calcualte reward
    noise = np.random.normal(0.0, 1.0)
    r_t = np.dot(a_t, theta_star) + noise 

    optimal_reward = max([np.dot(a, theta_star) for a in A_t])
    reg += optimal_reward - np.dot(a_t, theta_star)
    # print(reg/math.log(T))
    list_reg.append(reg/math.log(T))

    # save r_t and a_t
    list_y_t.append(a_t)
    list_r_t.append(r_t)

list_delta[1] = Lambda * np.eye(n_features)  + sum([np.outer(list_y_t[i], list_y_t[i]) for i in range(math.ceil(T_tleda[0]/2))])
delta1_inverse = np.linalg.inv(list_delta[1])
list_theta_hat[1] = np.dot(delta1_inverse, sum([list_r_t[i] * list_y_t[i] for i in range(math.ceil(T_tleda[0]/2))]))

Z = []
for i in range(math.ceil(T_tleda[0]/2), T_tleda[0]):
    Z_i = natural_elimination(list_A_t[i], list_delta[1], list_theta_hat[1])
    Z.append(Z_i)


# initialzing to be pi_2
pi_k = ExplorationPolicy(Z, T)
# print(pi_k)
end = False 
for k in range(1, M):
    for t in range(T_tleda[k-1]+1, T_tleda[k]):
        # observe X_t
        A_t = action_sets[np.random.randint(0,num_action_list)]
        # observe X_t
        list_A_t.append(A_t)
        # print('A_t chosen: ', A_t)

        A_t_k = []
        survivied_vector_set = []

        # find intersection set  
        for i in range(1, k+1):
            # print(M)
            # print('range', k)
            # print(i, list_theta_hat[i])
            survivied_vector_set.append(natural_elimination(list_A_t[i-1], list_delta[i], list_theta_hat[i]))
        # print(survivied_vector_set)
        print(survivied_vector_set)
        for cur_vector in survivied_vector_set[0]:
            # print('current vector', cur_vector)
            count = 0 
            for set in survivied_vector_set:
                for vector in set: 
                    if np.array_equal(vector, cur_vector):
                        count += 1 
                        break
            print('count', count)
            if count == len(survivied_vector_set):
                A_t_k.append(cur_vector)

        print(len(A_t_k))

        # play the arm with the feature vector y_t given pi_k(A_t_k)
        list_w, list_prob = pi_k
        selected_idx = np.random.choice(len(list_w), p=list_prob)
        # print(list_w)
        # print(selected_idx, len(list_w))
        W_j_inv = np.linalg.inv(list_w[selected_idx])
        #featured vector
        selected_idx = np.argmax([np.dot(np.dot(x.T, W_j_inv), x) for x in A_t_k])
        y_t = A_t_k[selected_idx]
        list_y_t.append(y_t)

        #receive a reward 
        noise = np.random.normal(0.0, 1.0)
        r_t = np.dot(y_t, theta_star) + noise
        list_r_t.append(r_t)

        # calculate optimal_reward
        optimal_reward = max([np.dot(a, theta_star) for a in A_t])

        reg += optimal_reward - np.dot(y_t, theta_star)
        # print('reg', reg/math.log(T))
 
        list_reg.append(reg/math.log(T))
        if t == T:
            end = True
            break

    if end:
        break
    # print(T_tleda)
    # print(list_T)
    # print(len(list_y_t))
    # print(k)
    # print('range', T_tleda[k-1]+1, T_tleda[k-1] + list_T[k]//2 + 1)

    # rangeT = min(T, T_tleda[k-1] + list_T[k]//2)
    delta_k = Lambda * np.eye(n_features) + sum([np.outer(list_y_t[i], list_y_t[i]) for i in range(T_tleda[k-1]+1, T_tleda[k-1] + list_T[k]//2)])
    list_delta[k+1] = delta_k
    delta_k_inverse = np.linalg.inv(delta_k)

    # print('sum', sum([list_r_t[i] * list_y_t[i] for i in range(T_tleda[k-1]+1, rangeT)]))
    list_theta_hat[k+1] =  np.dot(delta_k_inverse, sum([list_r_t[i] * list_y_t[i] for i in range(T_tleda[k-1]+1, T_tleda[k-1] + list_T[k]//2)]))
    # print(list_theta_hat[k+1])



    X_k_1 = []
    for t in range(T_tleda[k-1]+1, T_tleda[k-1] + T_tleda[k]//2):
        X_k_1_t = []
        survivied_vector_set2 = []
        for i in range(1, k+1):
            survivied_vector_set2.append(natural_elimination(list_A_t[i-1], list_delta[i], list_theta_hat[i]))
        
        for curr_vector in survivied_vector_set2[0]:
            count = 0
            for set in survivied_vector_set2:
                for vector in set:
                    if np.array_equal(vector, curr_vector):
                        count += 1
                        break
            if len(survivied_vector_set2) == count:
                X_k_1_t.append(curr_vector)
            
        X_k_1.append(X_k_1_t)
    print('here')
    pi_k = ExplorationPolicy(X_k_1, T)
    # print(pi_k)


print(reg/math.log(T))
print(list_theta_hat[-1])
print(theta_star)
# time = list(range(len(list_reg)))
# plt.plot(time, list_reg, label='Regret', color='blue')
# plt.xlabel('Time')
# plt.ylabel('Regret/sqrt T')
# plt.grid(True)

# # Save the figure before showing it
# plt.savefig(f'result/Context_Blind{T}_f-{n_features}_arms-{n_arms}_nal-{num_action_list}-_theta{n_theta}.png')

# Show the plot
plt.show()