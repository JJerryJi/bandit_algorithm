import numpy as np

def cartesian_to_spherical(x):
    r = np.linalg.norm(x)
    phi = []
    for i in range(len(x) - 2):
        value = np.arccos(x[i] / np.linalg.norm(x[i:]))
        # print('x', x[i:])
        # print(i, np.linalg.norm(x[i:]))
        phi.append(value)
        # print('phi', phi)

    # Calculate the last angle: phi[-1], where a range of (0, 2pi)
    if x[-1] >= 0:
        value = np.arccos(x[-2] / np.linalg.norm(x[-2:]))
        # print(x[-1], value)
    else:
        value = 2 * np.pi - np.arccos(x[-2] / np.linalg.norm(x[-2:]))
    phi.append(value)

    return r, phi

def spherical_to_cartesian(r, phi):
    ans = []
    for i in range(len(phi)):
        cur = r
        for j in range(i):
            cur *= np.sin(phi[j])
        cur *= np.cos(phi[i])
        ans.append(cur)
    
    last = r
    for i in range(len(phi)):
        last *= np.sin(phi[i])
    ans.append(last)
    
    return ans

# cur = [-1, 2, 4, 4]
# r, phi = cartesian_to_spherical(cur)

# print("Spherical:", phi)
# cartesian_result = spherical_to_cartesian(r, phi)
# print("Cartesian:", cartesian_result)
