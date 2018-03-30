from scipy.optimize import linprog
import numpy as np


def psi(n, x):
    if n==0: return 0.1
    elif n==1:
        if x == 0: return 0.1
        if x == 1: return 0.9
    elif n==2:
        if x == 0: return 0.9
        if x == 1: return 0.1

def psi_bin(xi, xj, beta=1.0):
    if xi == xj: return 0
    else: return beta



if __name__ == '__main__':
    c = [] # cost vector, i.e. maximise c^T*x
    b = [] # constraint result vector A*µ <= b, for equality: (A,A)^T*µ <= (b,b)^T

    n_pixels = 3
    n_labels = 2 # binary
    beta = -1.0

    # fill cost vector
    # unaries:
    for n in range(n_pixels):
        for label in range(n_labels):
            c.append(psi(n, label))

    # binaries:
    n_pairs = sum(range(n_pixels)) # how many pairs can be constructed (for binary potentials)?
    for pair in range(n_pairs):
        for label_i in range(n_labels):
            for label_j in range(n_labels):
                c.append(psi_bin(label_i, label_j, beta)) #TODO: This would have to be changed if psi_ij depended on i,j!


    # set up constraint matrix A and vector b:
    A = np.ndarray([0, len(c)])

    # unary conditions:
    for pixel in range(n_pixels):
        # each row in A (resp. entry in b) means a single condition
        row = np.zeros(len(c))
        for label in range(n_labels):
            row[2*pixel] = 1 # first condition: unaries µ_i have to sum to 1
            row[2*pixel+1] = 1  # first condition: unaries µ_i have to sum to 1
        A=np.vstack((A, row))
        b.append(1)

    # binary (pairwise) conditions:
    pairs = [[0, 1], [0, 2], [1, 2]] #TODO: Automatize this, e.g. using itertools.combinations (see http://stackoverflow.com/questions/5360220/how-to-split-a-list-into-pairs-in-all-possible-ways)
    pair = 0
    for i,j in pairs:
        # second condition: set all µ_ij to one along the first axis, subtract all µ_j in that axis, this should give 0
        for l in range(n_labels):
            row = np.zeros(len(c))
            for k in range(n_labels):
                row[n_labels*n_pixels + pair*2**n_labels + k*n_labels+l] = 1
            row[j*n_labels+l] = -1
            A = np.vstack((A, row))
            b.append(0)
        pair += 1

    pair = 0
    for i, j in pairs:
        # third condition: set all µ_ij to one along the second axis, subtract all µ_j in that axis, this should give 0
        for k in range(n_labels):
            row = np.zeros(len(c))
            for l in range(n_labels):
                row[n_labels*n_pixels + pair*2**n_labels + k*n_labels + l] = 1
            row[i*n_labels+k] = -1
            A = np.vstack((A, row))
            b.append(0)
        pair += 1



    # stack A matrix with -A, same for b vector, such that the inequality becomes an equality:
    A = np.vstack((A, -A))
    b_neg = [-x for x in b]
    b = b + b_neg

    print(A)

    # run linear program
    res = linprog(c, A_ub=A, b_ub=b, bounds=([0, 1]), options={"disp": True})
    print(res)