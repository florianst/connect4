import networkx as nx
import numpy as np

# function to draw an auxiliary graph:
def draw_graph(G, offset):
    import matplotlib.pyplot as plt

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > offset+0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= offset+0.5]
    pos=nx.shell_layout(G) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,pos,node_size=700)

    # edges
    nx.draw_networkx_edges(G,pos,edgelist=elarge, width=3)
    nx.draw_networkx_edges(G,pos,edgelist=esmall, width=3,alpha=0.5,edge_color='b',style='dashed')

    # labels
    nx.draw_networkx_labels(G,pos,font_size=10,font_family='sans-serif')

    plt.axis('off')
    plt.show() # display

# function to solve a 1D Potts MRF using Dijkstra's algorithm on an auxiliary graph
def solve_chain(n_variables, unaries_low, unaries_high, beta):
    n_states = 2 # TODO: Make this variable. Need to adjust generation of random unaries and their addition to the weight

    # list of random unaries unaries
    unaries = np.random.rand(n_states, n_variables)
    unaries = (unaries_high-unaries_low) * unaries + unaries_low # normalise
    unaries = np.append(unaries, 0)  # for last variable

    # Since our graph allows only paths of equal length from start to end, we can simply add a constant offset to all weights in order to cater for negative weights.
    # Seeing that both binary and unary factors are between -1 and 1, we can set this to 2.
    offset = 1-unaries_low

    # set up auxiliary graph using networkx
    G = nx.Graph()
    # set up entrance to graph
    for state_fin in range(n_states):
        G.add_edge('start', '0 ' + str(state_fin), weight=unaries[0] + offset)

    for i in range(n_variables - 1):
        for state_in in range(n_states):
            for state_fin in range(n_states):
                if (state_fin == 1): weight = unaries[i + 1] # include unary energies
                else: weight = 1-unaries[i+1]
                if (state_fin != state_in): weight = weight + beta # include binary (Potts) energies
                #print(str(i) + ' ' + str(state_in), str(i + 1) + ' ' + str(state_fin), weight)
                G.add_edge(str(i) + ' ' + str(state_in), str(i + 1) + ' ' + str(state_fin), weight=weight + offset)

    # set up exit from graph
    for state_in in range(n_states):
        G.add_edge(str(n_variables - 1) + ' ' + str(state_in), 'end', weight=0 + offset)  # we actually don't need the offset here

    #draw_graph(G, offset)

    # find shortest path through the auxiliary graph using Dijkstra's algorithm
    res = nx.dijkstra_path(G, 'start', 'end')

    # extract the useful information
    res = res[1:-1] # get rid of start and end elements
    res = [s[-1:] for s in res] # get rid of enumerating part in the variable names

    return res



print('unary energies in [0,1]:')
for beta in [0.01, 0.1, 0.2, 0.5, 1.0]:
    print ('beta = '+str(beta))
    print(solve_chain(n_variables=20, unaries_low=0, unaries_high=1, beta=beta))

print('unary energies in [-1,1]:')
for beta in [-1.0, -0.1, -0.01, 0.01, 0.1, 0.2, 0.5, 1.0]:
    print('beta = ' + str(beta))
    print(solve_chain(n_variables=20, unaries_low=-1, unaries_high=1, beta=beta))