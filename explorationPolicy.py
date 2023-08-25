import math 
import numpy as np 
n_features = 2

def ExplorationPolicy(Z, T):
    # initalization:
    k = 1 / (T * T)
    U = np.eye(n_features) * k
    n = 1 
    T_n = []
    W_n = U 
    m = len(Z)
    # define L here: 
    L = 1/(math.log(T * T * n_features))
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
        print('2*W_N Det', 2 * np.linalg.det(W_n))
        print('DET U', np.linalg.det(U) )
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


Z = [
    [np.array([2, 0.5]), np.array([1, 8]), np.array([3, 4])],
    [np.array([2, 0.5]), np.array([1, 8]), np.array([3, 4])],
    [np.array([2, 0.5]), np.array([1, 8]), np.array([3, 4])],
    [np.array([2, 0.5]), np.array([1, 8]), np.array([3, 4])],
    [np.array([1, 0])]
]
T = 10000

# Call the ExplorationPolicy function with the example inputs
print(Z)
list_W, list_T = ExplorationPolicy(Z, T)

# Print the resulting lists for verification
print("List of W matrices:")
for W in list_W:
    print(W)

print("List of T values:")
print(list_T)