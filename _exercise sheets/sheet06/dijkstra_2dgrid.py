import networkx as nx
import numpy as np


# function to solve a 1D Potts MRF using Dijkstra's algorithm on an auxiliary graph
def solve_chain(n_variables, n_states, unaries, beta):
    # list of random unaries unaries
    unaries = np.append(unaries, 0)  # for last variable

    # Since our graph allows only paths of equal length from start to end, we can simply add a constant offset to all weights in order to cater for negative weights.
    # Seeing that both binary and unary factors are between -1 and 1, we can set this to 2.
    offset = 1 - unaries_low

    # set up auxiliary graph using networkx
    G = nx.Graph()
    # set up entrance to graph
    for state_fin in range(n_states):
        G.add_edge('start', '0 ' + str(state_fin), weight=unaries[0] + offset)

    for i in range(n_variables - 1):
        for state_in in range(n_states):
            for state_fin in range(n_states):
                if (state_fin == 1):
                    weight = unaries[i + 1]  # include unary energies
                else:
                    weight = 1 - unaries[i + 1]
                if (state_fin != state_in): weight = weight + beta  # include binary (Potts) energies
                # print(str(i) + ' ' + str(state_in), str(i + 1) + ' ' + str(state_fin), weight)
                G.add_edge(str(i) + ' ' + str(state_in), str(i + 1) + ' ' + str(state_fin), weight=weight + offset)

    # set up exit from graph
    for state_in in range(n_states):
        G.add_edge(str(n_variables - 1) + ' ' + str(state_in), 'end',
                   weight=0 + offset)  # we actually don't need the offset here

    # find shortest path through the auxiliary graph using Dijkstra's algorithm
    res = nx.dijkstra_path(G, 'start', 'end')

    # extract the useful information
    res = res[1:-1]  # get rid of start and end elements
    res = [s[-1:] for s in res]  # get rid of enumerating part in the variable names

    return res

n = 20 # grid dimension: number of variables in each row/column

unaries_high =  1
unaries_low  = 0
beta = -0.5

# define random unaries (2D array)
unaries = np.random.rand(n, n)
unaries = (unaries_high-unaries_low) * unaries + unaries_low # normalise

# split unaries into two random variables
unaries_h = np.random.rand(n,n)
unaries = (unaries_high-unaries_low) * unaries_h + unaries_low # normalise

unaries_v = unaries-unaries_h # such that unaries = unaries_v + unaries_h


E_h = np.ndarray([0, n])
for i in range(n):
    E_h = np.vstack((E_h, solve_chain(n_variables=n, n_states=2, unaries=unaries_h[i,:], beta=beta)))

E_v = np.ndarray([0, n])
for i in range(n):
    E_v = np.vstack((E_v, solve_chain(n_variables=n, n_states=2, unaries=unaries_v[:,i], beta=beta)))

print('vertical variable values:', E_v)
print('horizontal variable values:', E_h)