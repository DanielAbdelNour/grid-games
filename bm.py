import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy
import time
from IPython import display

class BMBoard:
    def __init__(self, board_width=9):
        self.board_width = board_width
        self.board_shape = (self.board_width, self.board_width)
        self.action_direction = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]
        self.p1pos = (0,0)
        self.p2pos = (self.board_shape[0]-1, self.board_shape[0]-1)
        
        self.board = None
        self.bombs_board = None
        self.fire_board = None
        self.ammo_board = None
        self.powerup_board = None

        self._n_blocks = 20

        self._bomb_life = 4
        self._fire_life = 2

        self._start_health = 1
        self._start_ammo = 3
        self._start_power = 2

        self._players = [self.Entities.P1.value, self.Entities.P2.value]
        self._tickables = [self.Entities.BOMB.value, self.Entities.FIRE.value, self.Entities.AMMO.value, self.Entities.POWERUP.value]
        self._obstacles = [self.Entities.BLOCK.value, self.Entities.BOMB.value, self.Entities.P1.value, self.Entities.P2.value]
        self._obtainable = [self.Entities.POWERUP.value, self.Entities.AMMO.value]

        # player_id, health, ammo, power
        self.player_meta = np.array([
            [self.Entities.P1.value, self._start_health, self._start_ammo, self._start_power],
            [self.Entities.P2.value, self._start_health, self._start_ammo, self._start_power]
        ], dtype=np.int32)

        self.done = False
        self.restart_board()

        
    class Actions(Enum):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3
        BOMB = 4
        #DETONATE = 5
        NONE = 6

    class Entities(Enum):
        FLOOR = 0
        P1 = 1
        P2 = 2
        BLOCK = 3
        BOMB = 4
        FIRE = 5
        AMMO = 6
        POWERUP = 7


    def __repr__(self):
        return str(self.board + self.bombs_board + self.fire_board)

    def render(self, boards=None):
        cm = ListedColormap(["grey", "blue", "red", "saddlebrown", "black", "yellow"])

        if boards == None:
            eb = self.board.copy()
            bb = self.bombs_board.copy()
            fb = self.fire_board.copy()
        else:
            eb = boards[0].copy()
            bb = boards[1].copy()
            fb = boards[2].copy()

        eb[bb.nonzero()] = 4
        eb[fb.nonzero()] = 5
        plt.imshow(eb, cmap=cm, vmin=0, vmax=len(cm.colors))


    def restart_board(self):
        '''
        restart a board

        args
        p1pos, p2pos: tuple positions of players on board
        board_shape: tuple board dimensions
        '''
        self.board = np.zeros(self.board_shape, dtype=np.int32)
        self.board[self.p1pos[0], self.p1pos[1]] = self.Entities.P1.value
        self.board[self.p2pos[0], self.p2pos[1]] = self.Entities.P2.value

        # possible block positions
        possible_block_pos = np.argwhere(np.isin(self.board, [self.Entities.P1.value, self.Entities.P2.value]) == False)

        # choose and place N possible block positions
        block_pos_idxs = np.random.choice(np.arange(0, len(possible_block_pos)), self._n_blocks)
        for bpi in block_pos_idxs:
            bp = possible_block_pos[bpi]
            bpy, bpx = bp
            self.board[bpy, bpx] = self.Entities.BLOCK.value

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
            [self.Entities.P1.value, self._start_health, self._start_ammo, self._start_power],
            [self.Entities.P2.value, self._start_health, self._start_ammo, self._start_power]
        ], dtype=np.int32)

        self.done = False

        return self.board, self.bombs_board, self.fire_board, self.ammo_board, self.powerup_board, self.player_meta, self.done


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
        actions = []
        for a in list(self.Actions):
            # direction actions
            if a.value in [self.Actions.UP.value, self.Actions.DOWN.value, self.Actions.LEFT.value,self. Actions.RIGHT.value]:
                new_pos = player_pos + self.action_direction[a.value]
                py, px = new_pos
                # make sure new_pos is in bounds
                valid_bounds = np.all(new_pos >= 0) and np.all(new_pos < entity_board.shape[0])
                if valid_bounds:
                    valid_move = entity_board[py, px] == 0 and bombs_board[py, px] == 0
                    if valid_move:
                        actions.append(a.value)
            # bomb and detonation and check if player has enough ammo to place bomb
            if a.value in [self.Actions.BOMB.value]:
                py, px = player_pos
                valid_move = bombs_board[py, px] == 0 and player_meta[player_meta_idx, 2] > 0
                if valid_move:
                    actions.append(a.value)
        
        actions.append(self.Actions.NONE.value)

        return actions

            



    
    def add_bomb_to_board(self, board, pos, meta):
        board[pos[0], pos[1]] = self.Entities.BOMB.value
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
        if np.isin(entity_board[y, x], self._players):
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
        for dir in np.array(self.action_direction):
            # propagate the fire radius based on power
            for p in range(1, power):
                dy, dx = np.clip(pos + dir*p, a_min=0, a_max=self.board_width-1)
                # stop propagating if hit block
                if entity_board[dy, dx] == self.Entities.BLOCK.value:
                    break
                else:
                    board[dy, dx] = self._fire_life

        return board, player_meta


    def step(self, actions, simulate=False, prev_board=None, prev_bombs=None, prev_fire=None, prev_ammo=None, prev_powerup=None, prev_player_meta=None):
        '''
        apply supplied actions to a board state.
        actions are defined as a list of tuples, [(player_id, action_id), ...]
        '''
        if simulate == False:
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
            if a == self.Actions.NONE.value:
                continue

            # handle bombs
            if a == self.Actions.BOMB.value:
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
            if np.isin(board[afy, afx], self._players):
                # which idx in player meta?
                p = board[afy, afx]
                pm_idx = np.argwhere(player_meta[:,0] == p).ravel()[0]
                # apply damage
                player_meta[pm_idx, 1] -= 1

            fire_board[afy, afx] -= 1

        # check if any players have lost
        dead_players = player_meta[player_meta[:, 1] == 0, 0]
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
