# %%
from bm import BMBoard
import numpy as np
from copy import deepcopy
import time

# %%
class Node:
    def __init__(self, state, parent = None, player=None, action=None):
        self.state = state
        self.parent = parent
        self.player = player
        self.action = action
        self.visit_count = 0
        self.total_reward = 0
        self.uct = np.inf
        self.children = []
        self.fully_expanded = False

    def add_child(self, child):
        self.children.append(child)

    def update(self, reward):
        self.total_reward += reward
        self.visit_count += 1
        
    @property
    def value(self):
        return self.total_reward/self.visit_count if self.visit_count > 0 else np.inf

    @property
    def is_root(self):
        return self.parent == None

    @property
    def has_children(self):
        return len(self.children) > 0
    
# %%
# game = BMBoard(3)
# root = Node(game.board_state)

# %%
def run(game, root, n=1000):
    for _ in range(n):
        current_node = root

        # select best path
        while current_node.has_children and current_node.fully_expanded:
            children = current_node.children
            current_node = children[np.argmax([c.uct for c in children])]

        # which player is going to make an action
        player = 1 if current_node.player == None or current_node.player == 2 else 2

        # all valid actions
        valid_actions = game.valid_actions(player, current_node.state)
        actions_taken = [n.action for n in current_node.children if n.action != None]
        actions_available = list(set(valid_actions) - set(actions_taken))

        # reached end state of best path - break
        if len(actions_available) == 0:
            # check if the node will be fully expanded after this action is taken
            current_node.fully_expanded = set(valid_actions) == set(actions_taken)  
            continue

        # what actions have been taken from current_node 
        action = np.random.choice(actions_available)

        # take the action to generate a new state    
        if player == 1:
            new_state = game.step([(1, action), (2, BMBoard.Actions.NONE.value)], True, *current_node.state[:-1])
        else:
            new_state = game.step([(1, BMBoard.Actions.NONE.value), (2, action)], True, *current_node.state[:-1])

        # add the new node to its parent (current_node)
        new_node = Node(state=new_state, parent=current_node, player=player, action=action)
        current_node.add_child(new_node)

        # set current_node to new node for simulation
        current_node = new_node

        # make random actions until done
        _state = deepcopy(current_node.state)
        while _state[-1] == False:
            va1 = game.valid_actions(1, _state)
            va2 = game.valid_actions(2, _state)
            ra1 = np.random.choice(va1)
            ra2 = np.random.choice(va2)
            _state = game.step([(1, ra1), (2, ra2)], True, *_state[:-1])

        # get winner id
        meta = _state[-2]
        winner = meta[meta[:, 1] > 0, 0]
        winner_id = winner.item() if len(winner) == 1 else None

        # update parents
        parent = current_node
        while parent != None:
            reward = 1 if winner_id == parent.player else -1
            if winner_id == None: reward = 0
            parent.update(reward)
            parent = parent.parent

        # recalculate uct
        def update_uct(children):
            for c in children:
                c.uct = c.value + np.sqrt(2*np.log(c.parent.visit_count)/c.visit_count)
                if c.has_children:
                    update_uct(c.children)

        update_uct(root.children)

    best_child_idx = np.argmax([c.visit_count for c in root.children])
    best_child = root.children[best_child_idx]
    best_action = best_child.action
    return best_action

#%%

game = BMBoard(4, 1, 1000)
game.render()

#%%

root = Node(game.board_state)
best_action = run(game, root, 1000)
game.step([(1, best_action), (2, BMBoard.Actions.LEFT.value)])  
#print(game.board_state[-2])
game.render()


#%%
import time
import numpy as np
from bm_numba import BMBoard as BMBoard_Numba

#%%
game = BMBoard_Numba(4, 1, 1000)
game.render()
#%%
root = Node(game.board_state)
best_action = run(game, root, 1000)
game.step(np.array([[1, best_action], [2, BMBoard.Actions.LEFT.value]], dtype=np.int32))  
#print(game.board_state[-2])
game.render()