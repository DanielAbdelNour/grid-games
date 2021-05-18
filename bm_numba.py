import numpy as np
from enum import IntEnum
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy
import time
from IPython import display
import numba

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    BOMB = 4
    #DETONATE = 5
    NONE = 6

class Entities(IntEnum):
    FLOOR = 0
    P1 = 1
    P2 = 2
    BLOCK = 3
    BOMB = 4
    FIRE = 5
    AMMO = 6
    POWERUP = 7

@numba.extending.overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    def np_clip_impl(a, a_min, a_max, out=None):
        if out is None:
            out = np.empty_like(a)
        for i in np.arange(len(a)):
            if a[i] < a_min:
                out[i] = a_min
            elif a[i] > a_max:
                out[i] = a_max
            else:
                out[i] = a[i]
        return out
    return np_clip_impl

@numba.extending.overload(np.isin)
def np_isin(a, b):
    def np_isin_impl(a, b):
        shape = a.shape
        a = a.ravel()
        n = len(a)
        result = np.full(n, False)
        set_b = set(b)
        for i in np.arange(n):
            if a[i] in set_b:
                result[i] = True
        return result.reshape(shape)
    return np_isin_impl

spec = [
    ('board_width', numba.int32),
    ('board_shape', numba.int32[:]),
    ('action_direction', numba.int32[:, :]),
    ('p1pos', numba.int32[:]),
    ('p2pos', numba.int32[:]),

    ('board', numba.int32[:,:]),
    ('bombs_board', numba.int32[:,:]),
    ('fire_board', numba.int32[:,:]),
    ('ammo_board', numba.int32[:,:]),
    ('powerup_board', numba.int32[:,:]),

    ('_n_blocks', numba.int32),

    ('_bomb_life', numba.int32),
    ('_fire_life', numba.int32),

    ('_start_health', numba.int32),
    ('_start_ammo', numba.int32),
    ('_start_power', numba.int32),

    ('_players', numba.int32[:]),
    #('_tickables', numba.int32),
    #('_obstacles', numba.int32),
    #('_obtainable', numba.int32),

    ('player_meta', numba.int32[:,:]),

    ('done', numba.boolean)
]

@numba.experimental.jitclass(spec)
class BMBoardNumba:
    def __init__(self, board_width=9, start_health=3, start_ammo=3):
        self.board_width = board_width
        self.board_shape = np.array([self.board_width, self.board_width], dtype=np.int32)
        self.action_direction = np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
        ], dtype=np.int32)
        self.p1pos = np.array([0,0], dtype=np.int32)
        self.p2pos = np.array([self.board_shape[0]-1, self.board_shape[0]-1], dtype=np.int32)
        
        self.board = np.zeros((self.board_width, self.board_width), dtype=np.int32)
        self.bombs_board = np.zeros((self.board_width, self.board_width), dtype=np.int32)
        self.fire_board = np.zeros((self.board_width, self.board_width), dtype=np.int32)
        self.ammo_board = np.zeros((self.board_width, self.board_width), dtype=np.int32)
        self.powerup_board = np.zeros((self.board_width, self.board_width), dtype=np.int32)

        self._n_blocks = 20

        self._bomb_life = 4
        self._fire_life = 2

        self._start_health = start_health
        self._start_ammo = start_ammo
        self._start_power = 2

        self._players = np.array([1, 2], dtype=np.int32)
        #self._tickables = np.array([4, 5, 6, 7], dtype=np.int32)
        #self._obstacles = np.array([3, 4, 1, 2], dtype=np.int32)
        #self._obtainable = np.array([7, 6], dtype=np.int32)

        # player_id, health, ammo, power
        self.player_meta = np.array([
            [1, self._start_health, self._start_ammo, self._start_power],
            [2, self._start_health, self._start_ammo, self._start_power]
        ], dtype=np.int32)

        self.done = False
        self.restart_board()

    def __repr__(self):
        return str(self.board + self.bombs_board + self.fire_board)


    # def isin_nb(self, a, b):
    #     shape = a.shape
    #     a = a.ravel()
    #     n = len(a)
    #     result = np.full(n, False)
    #     set_b = set(b)
    #     for i in range(n):
    #         if a[i] in set_b:
    #             result[i] = True
    #     return result.reshape(shape)


    def render(self, boards=None):
        if boards == None:
            return self.board + self.fire_board + self.bombs_board
        else:
            return boards[0] + boards[1] + boards[2]

    def restart_board(self):
        '''
        restart a board

        args
        p1pos, p2pos: tuple positions of players on board
        board_shape: tuple board dimensions
        '''
        self.board = np.zeros((self.board_width, self.board_width), dtype=np.int32)
        self.board[self.p1pos[0], self.p1pos[1]] = Entities.P1.value
        self.board[self.p2pos[0], self.p2pos[1]] = Entities.P2.value

        # possible block positions
        possible_block_pos = np.argwhere(np.isin(self.board, np.array([Entities.P1.value, Entities.P2.value], dtype=np.int32)) == np.array(False, dtype=np.bool_))

        # choose and place N possible block positions
        block_pos_idxs = np.random.choice(np.arange(0, len(possible_block_pos)), self._n_blocks)
        for bpi in block_pos_idxs:
            bp = possible_block_pos[bpi]
            bpy, bpx = bp
            self.board[bpy, bpx] = Entities.BLOCK.value

        # clear positions adjacent to players
        for y,x in np.array([[0,1], [1,0], [1,1]], dtype=np.int32):
            self.board[y, x] = 0
            self.board[self.board_width-1-y, self.board_width-1-x] = 0
        
        self.bombs_board = np.zeros_like(self.board)
        self.fire_board = np.zeros_like(self.board)
        self.ammo_board = np.zeros_like(self.board)
        self.powerup_board = np.zeros_like(self.board)

        # player_id, health, ammo, power
        self.player_meta = np.array([
            [Entities.P1.value, self._start_health, self._start_ammo, self._start_power],
            [Entities.P2.value, self._start_health, self._start_ammo, self._start_power]
        ], dtype=np.int32)

        self.done = False

        return self.board, self.bombs_board, self.fire_board, self.ammo_board, self.powerup_board, self.player_meta, self.done

    @property
    def board_state(self):
        return self.board, self.bombs_board, self.fire_board, self.ammo_board, self.powerup_board, self.player_meta, self.done
    
    def valid_actions(self, player_id, board_states):
        '''
        returns a list of valid actions from the given board state
        '''
        current_board_states = board_states
        entity_board, bombs_board, fire_board, ammo_board, powerup_board, player_meta, done = current_board_states
        player_pos = np.argwhere(entity_board == player_id)[0]
        player_meta_idx = np.argwhere(player_meta[:, 0] == player_id).item()
        # check actions
        actions = np.empty(0, dtype=np.int32)
        for a in np.array([0,1,2,3,4,6], dtype=np.int32): # cant iteratve via Actions directly with numba
            # direction actions
            if np.isin(np.array([a], dtype=np.int32), np.array([Actions.UP.value, Actions.DOWN.value, Actions.LEFT.value, Actions.RIGHT.value])):
                new_pos = player_pos + self.action_direction[a]
                py, px = new_pos
                # make sure new_pos is in bounds
                valid_bounds = np.all(new_pos >= 0) and np.all(new_pos < entity_board.shape[0])
                if valid_bounds:
                    valid_move = entity_board[py, px] == 0 and bombs_board[py, px] == 0
                    if valid_move:
                        actions = np.append(actions, a)
            # bomb and detonation and check if player has enough ammo to place bomb
            if a in np.array([Actions.BOMB.value]):
                py, px = player_pos
                valid_move = bombs_board[py, px] == 0 and player_meta[player_meta_idx, 2] > 0
                if valid_move:
                    actions = np.append(actions, a)
        
        actions = np.append(actions, Actions.NONE.value)

        return actions

    def add_bomb_to_board(self, board, pos, meta):
        board[pos[0], pos[1]] = Entities.BOMB.value
        meta = np.append(meta, [pos, self._bomb_life])
        return board, meta

    
    def player_meta_to_dict(self, meta):
        id, hp, ammo, power = meta
        meta_dict = {
            'id': id,
            'hp':hp,
            'ammo':ammo,
            'power':power
        }
        return meta_dict


    def add_bomb(self, board, pos, entity_board, player_meta):
        '''
        add bomb to bomb board at position
        requires entity_board to check for blocks and players
        requires player_meta to register ammo changes
        '''
        board = board.copy()
        player_meta = player_meta.copy()
        y, x = pos

        # ensure not placed ontop of another bomb
        if board[y,x] != 0:
            return board, player_meta

        board[y, x] = self._bomb_life

        # decrease ammo
        if np.isin(np.array(entity_board[y, x], dtype=np.int32), self._players):
            # which idx in player meta?
            p = entity_board[y, x]
            pm_idx = np.argwhere(player_meta[:,0] == p).ravel()[0]
            # apply ammo change
            player_meta[pm_idx, 2] -= 1

        return board, player_meta        



    def add_fire(self, board, pos, entity_board, player_meta):
        '''
        add fire to fire board at position
        requires entity_board to check for blocks and players
        requires player_meta to register damage
        '''
        power = 2
        board = board.copy()
        player_meta = player_meta.copy()
        y, x = pos
        board[y, x] = self._fire_life
        # add fire entity in 4 directions from the center
        for dir in self.action_direction:
            # propagate the fire radius based on power
            for p in np.arange(1, power):
                dy, dx = np.clip(pos + dir*p, a_min=0, a_max=self.board_width-1)
                # stop propagating if hit block
                if entity_board[dy, dx] == Entities.BLOCK.value:
                    break
                else:
                    board[dy, dx] = self._fire_life

        return board, player_meta

    
    def step(
        self, 
        actions, 
        simulate=False, 
        prev_board=np.empty((9,9), dtype=np.int32), 
        prev_bombs=np.empty((9,9), dtype=np.int32), 
        prev_fire=np.empty((9,9), dtype=np.int32), 
        prev_ammo=np.empty((9,9), dtype=np.int32), 
        prev_powerup=np.empty((9,9), dtype=np.int32), 
        prev_player_meta=np.empty((9,9), dtype=np.int32)):
        '''
        apply supplied actions to a board state.
        actions are defined as a list of tuples, [(player_id, action_id), ...]
        '''
        if simulate == np.array(False, dtype=np.bool_):
            board = self.board.copy()
            bombs_board = self.bombs_board.copy()
            fire_board = self.fire_board.copy()
            ammo_board = self.ammo_board.copy()
            powerup_board = self.powerup_board.copy()
            player_meta = self.player_meta.copy()
        else:
            board = prev_board.copy()
            bombs_board = prev_bombs.copy()
            fire_board = prev_fire.copy()
            ammo_board = prev_ammo.copy()
            powerup_board = prev_powerup.copy()
            player_meta = prev_player_meta.copy()

        done = self.done

        # apply actions to layers
        for i, pa in enumerate(actions):
            # player id, action value
            p, a = pa

            p_ammo = player_meta[player_meta[:, 0] == p, 2][0]
            p_currpos = np.argwhere(board == p)[0]

            # handle none
            if a == Actions.NONE.value:
                continue

            # handle bombs
            if a == Actions.BOMB.value:
                if p_ammo <= 0:
                    continue
                bombs_board, player_meta = self.add_bomb(bombs_board, p_currpos, board, player_meta)
                continue
            
            # bounded proposed new pos
            p_newpos = np.clip(p_currpos + self.action_direction[a], a_min=0, a_max=self.board_width-1)
            
            # apply movement to player board
            if board[p_newpos[0], p_newpos[1]] == 0 and bombs_board[p_newpos[0], p_newpos[1]] == 0:
                board[p_currpos[0], p_currpos[1]] = 0
                board[p_newpos[0], p_newpos[1]] = p

        # tick bombs
        active_bombs = np.argwhere(bombs_board > 0)
        for ab in active_bombs:
            bombs_board[ab[0], ab[1]] -= 1
            # explode bombs
            if bombs_board[ab[0], ab[1]] == 0:
                # add fire
                fire_board, player_meta = self.add_fire(fire_board, ab, board, player_meta)

        
        # chain bombs
        active_fire = np.argwhere(fire_board > 0)
        bi = 0
        while bi < len(active_fire):
            af = active_fire[bi]
            afy, afx = af
            if bombs_board[afy, afx] > 0:
                bombs_board[afy, afx] = 0
                fire_board, player_meta = self.add_fire(fire_board, af, board, player_meta)
            active_fire = np.argwhere(fire_board > 0)
            bi += 1
        
        
        # tick fire
        active_fire = np.argwhere(fire_board > 0)
        for af in active_fire:
            afy, afx = af

            # apply damage to players
            if np.isin(np.array(board[afy, afx], dtype=np.int32), self._players):
                # which idx in player meta?
                p = board[afy, afx]
                pm_idx = np.argwhere(player_meta[:,0] == p).ravel()[0]
                # apply damage
                player_meta[pm_idx, 1] -= 1

            fire_board[afy, afx] -= 1

        # check if any players have lost
        dead_players = player_meta[player_meta[:, 1] <= 0, 0]
        if len(dead_players) > 0:
            done = True

        # modify internal board states if not simulating, otherwise return the modified board states
        if simulate == False:
            self.board = board
            self.bombs_board = bombs_board
            self.fire_board = fire_board
            self.ammo_board = ammo_board
            self.powerup_board = powerup_board
            self.player_meta = player_meta
            self.done = done
        else:
            return board, bombs_board, fire_board, ammo_board, powerup_board, player_meta, done
