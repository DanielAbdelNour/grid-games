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
    def __init__(self, state, parent=None, children=[], player=1, chosen_action=None):
        self.id = id(self)
        self.state = state
        self.parent = parent
        self.children = children
        self.is_terminal = False
        self.player = player # id of the player (1 or 2)

        self.score_switch = 1 if self.player == 1 else -1

        self.visit_count = 0
        self.value_sum = 0
        self.ucb1_score = np.inf

        self.chosen_action = chosen_action

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

#%%
b = BMBoard()
board_state = b.restart_board()

#%%
board_state = b.board_state()
root = Node(board_state)
possible_actions = [a.value for a in list(b.Actions)]
maximiser_id = 1
minimiser_id = 2

# expand the tree on init
for i, a in enumerate(possible_actions):
    min_max_action = [(1, a), (2, BMBoard.Actions.NONE.value)]
    _board_state = b.step(min_max_action, True, *board_state[:-1])
    new_node = Node(_board_state, root, [], 1, min_max_action)
    root.add_child(new_node)

# keep track of current maximiser
current_player = maximiser_id

for i in range(1000):
    current_node = root
    
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
            child_scores = np.array([cn.ucb1_score for cn in non_terminal_children], dtype=np.float64)
            current_node = non_terminal_children[np.argmax(child_scores)]
            has_children = len(current_node.children) > 0


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

            _board_state = b.step(min_max_action, True, *board_state[:-1])
            new_node = Node(_board_state, current_node, [], current_player, min_max_action)
            # set new node to terminal if the state is done
            if _board_state[-1] == True:
                new_node.is_terminal = True
            
            # add new node as child of current node
            current_node.add_child(new_node)

        # select a new child to be current for simulation from valid children (non terminal)
        current_node = np.random.choice([c for c in current_node.children if c.is_terminal == False])


    '''simulate'''
    r_board_state = deepcopy(current_node.state)
    r_done = r_board_state[-1]
    while r_done == False:
        # random actions
        ra1 = np.random.choice(possible_actions)
        ra2 = np.random.choice(possible_actions)
        r_board_state = b.step([(1, ra1), (2, ra2)], True, *r_board_state[:-1])
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
        next_node.value_sum += reward * 1 if next_node.player == minimiser_id else -1
        next_node = next_node.parent

    # update ucb for all nodes
    def update_ucb(node):   
        for child in node.children:
            if len(child.children) > 0:
                update_ucb(child)
            child.ucb1_score = UCB1(child.value(), 2, root.visit_count, child.visit_count) 

    root.ucb1_score = UCB1(root.value(), 2, root.visit_count, root.visit_count)
    update_ucb(root)  

    #print(np.round((i/1000) * 100))

# %%
def run(game, n=1000):
    board_state = game.board_state()
    root = Node(board_state)
    possible_actions = [a.value for a in list(game.Actions)]
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
                child_scores = np.array([cn.ucb1_score for cn in non_terminal_children], dtype=np.float64)
                current_node = non_terminal_children[np.argmax(child_scores)]
                has_children = len(current_node.children) > 0


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
            current_node = np.random.choice([c for c in current_node.children if c.is_terminal == False])


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
            next_node.value_sum += reward * 1 if next_node.player == minimiser_id else -1
            next_node = next_node.parent

        # update ucb for all nodes
        def update_ucb(node):   
            for child in node.children:
                if len(child.children) > 0:
                    update_ucb(child)
                child.ucb1_score = UCB1(child.value(), 2, root.visit_count, child.visit_count) 

        root.ucb1_score = UCB1(root.value(), 2, root.visit_count, root.visit_count)
        update_ucb(root)  

        if i % 100 == 0:
            print(i)

    best_action = root.children[np.argmax([c.visit_count for c in root.children])].chosen_action[0][1]
    return best_action

#%%
b = BMBoard()
board_state = b.restart_board()

for i in range(10):
    ba = run(b, 1000)
    b.step([(1, ba), (2, BMBoard.Actions.NONE.value)])
# %%
