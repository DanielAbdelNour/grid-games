# %%
from logging import debug
from bm import BMBoard
from typing import Tuple

from numba.typed import typedlist
from bm_numba import BMBoardNumba, Actions, Entities
import numpy as np
from copy import deepcopy
import numba
from numba import types, deferred_type, optional, typed

# %%
node_type = deferred_type()

spec = [
    ('state', types.Tuple([
        types.Array(numba.int_, 2, 'C'),
        types.Array(numba.int_, 2, 'C'),
        types.Array(numba.int_, 2, 'C'),
        types.Array(numba.int_, 2, 'C'),
        types.Array(numba.int_, 2, 'C'),
        types.Array(numba.int_, 2, 'C'),
        numba.boolean
    ])),
    ('parent', optional(node_type)),
    ('player', optional(numba.int_)),
    ('action', numba.int_),
    ('visit_count', numba.int_),
    ('total_reward', numba.int_),
    ('uct', numba.int_),
    ('left_child', optional(node_type)),
    ('right_child', optional(node_type)),
    ('sibling', optional(node_type)),
    ('fully_expanded', numba.boolean)
]

@numba.experimental.jitclass(spec)
class Node:
    def __init__(self, state, parent=None, player=None, action=999):
        self.state = state
        self.parent = parent
        self.player = player
        self.action = action
        self.visit_count = 0
        self.total_reward = 0
        self.uct = 999999
        self.left_child = None
        self.right_child = None
        self.sibling = None
        self.fully_expanded = False
        if parent is not None:
            parent.add_child(self)

    def update(self, reward):
        self.total_reward += reward
        self.visit_count += 1

    def add_sibling(self, sibling):
        self.sibling = sibling

    def add_child(self, child):
        child.parent = self
        # if parent has no children
        if self.left_child is None:
            # this node is it first child
            self.left_child = child
            self.right_child = child
        else:
            # the last (now right) child will have this node as sibling
            self.right_child.add_sibling(child)
            self.right_child = child

    @property
    def children(self):
        """ A list with all children. """
        children = []
        child = self.left_child
        while child is not None:
            children.append(child)
            child = child.sibling
        return children

    @property
    def value(self):
        return self.total_reward/self.visit_count if self.visit_count > 0 else 999999

    @property
    def is_root(self):
        return self.parent is None

    @property
    def has_children(self):
        return len(self.children) > 0
    
node_type.define(Node.class_type.instance_type)

#%%
game = BMBoardNumba(4, 1, 1000)
root = Node(game.board_state)
print(root.is_root)
game.render()

#%%
root.add_child(Node(game.board_state))
root.add_child(Node(game.board_state))
root.children


# %%
def update_uct(children):
    for c in children:
        c.uct = c.value + np.sqrt(2*np.log(c.parent.visit_count)/c.visit_count)
        if c.has_children:
            update_uct(c.children)

#%%
@numba.njit
def update_uct2(root):
    current = root
    stack = []
    all_nodes = []

    while True:
        for child in current.children:
            all_nodes.append(child)
            stack.append(child)
        if len(stack) > 0:
            current = stack.pop()
        else:
            break

    return all_nodes


#%%
#@numba.jit
def run(game, root, n=1000):
    for i in np.arange(n):
        current_node = root

        # select best path
        while current_node.has_children and current_node.fully_expanded:
            children = current_node.children
            current_node = children[np.argmax(np.array([c.uct for c in children], dtype=np.int_))]

        # which player is going to make an action
        player = 1 if current_node.player == None or current_node.player == 2 else 2

        # all valid actions
        valid_actions = game.valid_actions(player, current_node.state)
        
        actions_taken = np.array([n.action for n in current_node.children if n.action != 999], dtype=np.int_)
        actions_available = np.array(list(set(valid_actions) - set(actions_taken)), dtype=np.int_)

        # reached end state of best path - break
        if len(actions_available) == 0:
            # check if the node will be fully expanded after this action is taken
            current_node.fully_expanded = set(valid_actions) == set(actions_taken)  
            continue

        # what actions have been taken from current_node 
        action = np.random.choice(actions_available)

        # take the action to generate a new state    
        if player == 1:
            new_state = game.step(np.array([(1, action), (2, BMBoard.Actions.NONE.value)], np.int_), True, *current_node.state[:-1])
        else:
            new_state = game.step(np.array([(1, BMBoard.Actions.NONE.value), (2, action)], np.int_), True, *current_node.state[:-1])

        # add the new node to its parent (current_node)
        new_node = Node(state=new_state, parent=current_node, player=player, action=action)
        current_node.add_child(new_node)

        # set current_node to new node for simulation
        current_node = new_node

        # make random actions until done
        _state = deepcopy(current_node.state)
        _iter = 0
        while _state[-1] == False:
            va1 = game.valid_actions(1, _state)
            va2 = game.valid_actions(2, _state)
            ra1 = np.random.choice(va1)
            ra2 = np.random.choice(va2)
            _state = game.step(np.array([(1, ra1), (2, ra2)], np.int_), True, *_state[:-1])
            _iter += 1
            if _iter > 1000:
                print('stuck')
                break

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

        
        #recalculate uct
        all_nodes = update_uct2(root)
        for c in all_nodes:
            c.uct = c.value + np.sqrt(2*np.log(c.parent.visit_count)/c.visit_count)

    best_child_idx = np.argmax(np.array([c.visit_count for c in root.children], dtype=np.int_))
    best_child = root.children[best_child_idx]
    best_action = best_child.action
    return best_action


#%%
game = BMBoardNumba(4, 1, 1000)
#%%
root = Node(game.board_state)
best_action = run(game, root, 1000)
game.step(np.array([[1, best_action], [2, Actions.LEFT.value]], dtype=np.int_))  
print(game.board_state[-2])
game.render()
# %%

# %%
