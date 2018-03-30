from scipy import optimize
import numpy as np
import copy


def phiP(v1, v2, beta=1):
    if v1 == v2:
        return 0
    else:
        return beta

noLabels = 2

# create structure to get linear index by calling elementIndex[i,j,k,l]
# unaries are treated as i = j, and k = l
pairwiseIndex = np.empty((3,3,2,2))
pairwiseIndex[:] = np.NAN
unaryIndex = copy.deepcopy(pairwiseIndex)

# setup c
c = np.ndarray([0])

# hard coding of unaries
# i, j, k, l
c = np.append(c, 0.1)
unaryIndex[0, 0, 0, 0] = c.size - 1

c = np.append(c, 0.1)
unaryIndex[0, 0, 1, 1] = c.size - 1

c = np.append(c, 0.1)
unaryIndex[1, 1, 0, 0] = c.size - 1

c = np.append(c, 0.9)
unaryIndex[1, 1, 1, 1] = c.size - 1

c = np.append(c, 0.9)
unaryIndex[2, 2, 0, 0] = c.size - 1

c = np.append(c, 0.1)
unaryIndex[2, 2, 1, 1] = c.size - 1

noUnaries = int(c.size / noLabels)
unaryIndex = unaryIndex.astype(int)

def insertPairwise(c, pairwiseIndex, i, j):
    for k in range(noLabels):
        for l in range(noLabels):
            c = np.append(c, phiP(k, l))
            pairwiseIndex[i, j, k, l] = c.size - 1
            pairwiseIndex = pairwiseIndex.astype(int)
    return c, pairwiseIndex

# append pairwise
pairwiseTerms = [[0,1], [0,2], [1,2]]
for i,j in pairwiseTerms:
    c, pairwiseIndex = insertPairwise(c, pairwiseIndex, i, j)

# set conditions
b = np.ndarray([0])
A = np.ndarray([0, c.size])

# unary conditions
for sign in [1, -1]:
    for i in range(noUnaries):
        conditionVector = np.zeros([c.size])
        for k in range(noLabels):
            conditionVector[unaryIndex[i,i,k,k]] = sign
        A = np.vstack((A, conditionVector))
        b = np.append(b, sign)


# pairwise conditions
for sign in [1,-1]:
    for i,j in pairwiseTerms:
        for k in range(noLabels):
            conditionVector = np.zeros([c.size])
            for l in range(noLabels):
                conditionVector[pairwiseIndex[i,j,k,l]] = sign
            conditionVector[unaryIndex[i, i, k, k]] = -sign
            A = np.vstack((A, conditionVector))
            b = np.append(b, 0)

for sign in [1,-1]:
    for i,j in pairwiseTerms:
        for l in range(noLabels):
            conditionVector = np.zeros([c.size])
            for k in range(noLabels):
                conditionVector[pairwiseIndex[i,j,k,l]] = sign
            conditionVector[unaryIndex[j, j, l, l]] = -sign
            A = np.vstack((A, conditionVector))
            b = np.append(b, 0)



print(A)
#print(A)
#print(b)


#res = optimize.linprog(c, A_ub = A, b_ub = b, bounds = ([0,1]), options = {"disp": True})
#print(res)
