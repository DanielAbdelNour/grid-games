#%%
import time
from IPython import display
from bm import BMBoard
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy

#%%
class Node:
    def __init__(self, state, parent=None, player=1, chosen_action=None):
        self.id = id(self)
        self.state = state
        self.parent = parent
        self.children = []
        self.is_terminal = False
        self.player = player # id of the player (1 or 2)

        self.score_switch = 1 if self.player == 1 else -1

        self.visit_count = 0
        self.value_sum = 0

        self.uct = np.inf

        self.chosen_action = chosen_action
        self.is_root = True if self.parent == None else False

    def add_child(self, node):
        self.children = np.append(self.children, node)

    def value(self):
        if self.visit_count == 0:
            return np.inf
        return self.value_sum/self.visit_count

    def __repr__(self):
        return str({key:value for key, value in Node.__dict__.items() if not key.startswith('__') and not callable(key)})

    def render(self):
        cm = ListedColormap(["grey", "blue", "red", "saddlebrown", "black", "yellow"])
        eb = self.state[0].copy()
        bb = self.state[1].copy()
        fb = self.state[3].copy()
        eb[bb.nonzero()] = 4
        eb[fb.nonzero()] = 5
        plt.imshow(eb, cmap=cm, vmin=0, vmax=len(cm.colors))
        
#%%
def UCB1(value, c, N, n):
    if n == 0:
        return np.inf
    return value + c * np.sqrt(np.log(N)/n)

# %%
def update_ucb(node, N, C=1):   
    for child in node.children:
        if len(child.children) > 0:
            update_ucb(child, N)
        child.ucb1_score = UCB1(child.value(), C, N, child.visit_count) 


#%%
b = BMBoard(3)
b.render()

# %%
ba, root_node = run2(b, 50000, C=2)
b.step([(1, ba), (2, BMBoard.Actions.DOWN.value)])
print(b.player_meta)
b.render()

#%%
from pyvis.network import Network
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd


#%%
node_net = []

def build_net(node):
    global node_net
    for child in node.children:
        node_meta = {'source':node.id, 'target': child.id}
        node_net.append(node_meta)
        if len(child.children) > 0:
            build_net(child)

build_net(root_node)    
net_df = pd.DataFrame(node_net)

#%%
G = nx.from_pandas_edgelist(net_df, source='source', target='target')

# %%
nt = Network('500px', '500px', layout=True)
nt.from_nx(G)
nt.show_buttons()
nt.toggle_physics(False)
nt.save_graph('graph.html')
# %%
