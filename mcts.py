#%%
import time
from IPython import display
from numba.core.decorators import njit
from bm import BMBoard
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy

#%%
#b = BMBoard()

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
        self.ucb1_score = np.inf

        self.chosen_action = chosen_action
        self.is_root = True if self.parent == None else False

    def add_child(self, node):
        self.children = np.append(self.children, node)

    def value(self):
        if self.visit_count == 0:
            return np.inf
        return self.value_sum/self.visit_count

    def __repr__(self):
        return str({
            'id': self.id,
            'player': self.player,
            'chosen_action': self.chosen_action,
            'visit_count': self.visit_count,
            'value_sum': self.value_sum,
            'ucb1_score': self.ucb1_score
        })

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
def run(game, n=1000, C=1):
    board_state = game.board_state()
    root = Node(board_state)
    possible_actions = [0,1,2,3,4] #[a.value for a in list(game.Actions)]
    maximiser_id = 1
    minimiser_id = 2

    # expand the tree on init
    for i, a in enumerate(possible_actions):
        min_max_action = [(1, a), (2, BMBoard.Actions.NONE.value)]
        _board_state = game.step(min_max_action, True, *board_state[:-1])
        new_node = Node(_board_state, root, [], 1, min_max_action)
        root.add_child(new_node)

    # keep track of current maximiser
    current_player = maximiser_id

    for i in range(n):
        current_node = root
        terminate = False
        
        '''selection'''
        # get the child values from the current node
        visits = np.array([c.visit_count for c in current_node.children], dtype=np.int64)

        # check if no children have been visited
        if np.all(visits == 0):
            # select a random child
            current_node = np.random.choice([c for c in current_node.children if c.is_terminal == False])
        else:
            # select the child with highest UCB1 score
            has_children = True
            while has_children == True:
                # change current node to child with highest ucb1 that isn't terminal
                non_terminal_children = [c for c in current_node.children if c.is_terminal == False]
                if len(non_terminal_children) == 0:
                    terminate = True
                    break
                child_scores = np.array([cn.ucb1_score for cn in non_terminal_children], dtype=np.float64)
                current_node = non_terminal_children[np.argmax(child_scores)]
                has_children = len(current_node.children) > 0

        if terminate:
            break

        '''expansion'''
        # check if the node is a leaf node
        if current_node.visit_count > 0 and len(current_node.children) == 0:
            # new level being added, so flip player
            current_player = maximiser_id if current_player == minimiser_id else minimiser_id
            board_state = deepcopy(current_node.state)
            for a in possible_actions:
                if current_player == maximiser_id:
                    min_max_action = [(1, a), (2, BMBoard.Actions.NONE.value)]
                else:
                    min_max_action = [(1, BMBoard.Actions.NONE.value), (2, a)]

                _board_state = game.step(min_max_action, True, *board_state[:-1])
                new_node = Node(_board_state, current_node, [], current_player, min_max_action)
                # set new node to terminal if the state is done
                if _board_state[-1] == True:
                    new_node.is_terminal = True
                
                # add new node as child of current node
                current_node.add_child(new_node)

            # select a new child to be current for simulation from valid children (non terminal)
            non_terminal_children = [c for c in current_node.children if c.is_terminal == False]
            if len(non_terminal_children) == 0:
                continue

            current_node = np.random.choice(non_terminal_children)

        '''simulate'''
        r_board_state = deepcopy(current_node.state)
        r_done = r_board_state[-1]
        while r_done == False:
            # random actions
            ra1 = np.random.choice(possible_actions)
            ra2 = np.random.choice(possible_actions)
            r_board_state = game.step([(1, ra1), (2, ra2)], True, *r_board_state[:-1])
            r_done = r_board_state[-1]

        # get the score after rollout
        r_meta = r_board_state[-2]
        winner = r_meta[r_meta[:, 1] != 0, 0].ravel()
        if len(winner) == 0:
            # tied
            reward = 0.5
        elif winner[0] == current_player:
            reward = 1
        else:
            reward = -1

        '''backprop'''
        # update n_visits and t_reward up chain
        next_node = current_node
        while next_node != None:
            # switch reward based on node player
            next_node.visit_count += 1
            next_node.value_sum += reward * 1 if next_node.player == maximiser_id else -1
            next_node = next_node.parent

        #root.ucb1_score = UCB1(root.value(), 2, root.visit_count, root.visit_count)
        update_ucb(root, N = root.visit_count, C=C)  

    #best_action = root.children[np.argmax([c.visit_count for c in root.children])].chosen_action[0][1]
    best_action = root.children[np.argmax([c.ucb1_score for c in root.children])].chosen_action[0][1]
    return best_action, root

#%%
def run2(game, n=1000, C=1):
    board_state = game.board_state()
    root = Node(board_state)
    has_children = False
    maximiser_id = 1
    minimiser_id = 2

    for i in range(100):
        current_node = root
        # select best path
        has_children = len(current_node.children) > 0
        while has_children:
            # ensure current nodes has had all valid actions expanded
            valid_actions = game.valid_actions(current_node.player, current_node.state)
            explored_actions = [n.chosen_action for n in current_node.children]
            # do not continue selection if there are unexplored actions
            if np.all(np.isin(np.array(valid_actions), np.array(explored_actions))) == False:
                break
            # do not continue if all children are terminal
            non_terminal_children = [c for c in current_node.children if c.is_terminal == False]
            if len(non_terminal_children) == 0:
                break
            child_scores = np.array([cn.ucb1_score for cn in non_terminal_children], dtype=np.float64)
            current_node = non_terminal_children[np.argmax(child_scores)]
            has_children = len(current_node.children) > 0

        # expand unboserved
        # who is about to play
        if current_node.is_root:
            player = maximiser_id
        else:
            player = minimiser_id if current_node.player == maximiser_id else maximiser_id
        valid_actions = np.array(game.valid_actions(player, current_node.state))
        explored_actions = np.array([n.chosen_action for n in current_node.children])
        unexplored_action_idx = np.argwhere(np.isin(valid_actions, explored_actions) == False).ravel()
        if len(unexplored_action_idx) == 0:
            current_node = np.random.choice(current_node.children)
        else:
            rand_action = np.random.choice(valid_actions[unexplored_action_idx])
            if player == maximiser_id:
                min_max_action = [(1, rand_action), (2, BMBoard.Actions.NONE.value)]
            else:
                min_max_action = [(1, BMBoard.Actions.NONE.value), (2, rand_action)]
            _board_state = game.step(min_max_action, True, *current_node.state[:-1])
            new_node = Node(_board_state, current_node, player, rand_action)
            if _board_state[-1] == True:
                new_node.is_terminal = True
            current_node.add_child(new_node)
            current_node = new_node


        # simulate
        r_board_state = deepcopy(current_node.state)
        r_done = r_board_state[-1]
        while r_done == False:
            # random actions
            ra1 = np.random.choice(game.valid_actions(1, r_board_state))
            ra2 = np.random.choice(game.valid_actions(2, r_board_state))
            r_board_state = game.step([(1, ra1), (2, ra2)], True, *r_board_state[:-1])
            r_done = r_board_state[-1]

        # get the score after rollout
        r_meta = r_board_state[-2]
        winner = r_meta[r_meta[:, 1] != 0, 0].ravel()
        if len(winner) == 0:
            # tied
            reward = 0.5
        elif winner[0] == player:
            reward = 1
        else:
            reward = -1

        # backprop
        # update n_visits and t_reward up chain
        next_node = current_node
        while next_node != None:
            # switch reward based on node player
            next_node.visit_count += 1
            next_node.value_sum += reward * 1 if next_node.player == maximiser_id else -1
            next_node = next_node.parent

        #root.ucb1_score = UCB1(root.value(), 2, root.visit_count, root.visit_count)
        update_ucb(root, N = root.visit_count, C=C)  

    #best_action = root.children[np.argmax([c.visit_count for c in root.children])].chosen_action[0][1]
    best_action = root.children[np.argmax([c.ucb1_score for c in root.children])].chosen_action
    return best_action, root

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
