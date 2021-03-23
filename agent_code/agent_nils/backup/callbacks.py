import os
import pickle
import random

import numpy as np
import sys
sys.path.append('/home/nils/Documents/ifml_project/bomberman/agent_code/rule_based_agent')
from callbacks_ import setup as setup_
from callbacks_ import act as act_
from collections import deque
import time

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def model_evaluation(model, features):
    current = model
    for f in features:
        current = current[f]
    return current

def setup(self):
    #setup_(self)

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros((6, 6, 6, 2, 6))

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

def act(self, game_state: dict) -> str:

    if not game_state:
        return 'WAIT'

    # hab ich gesetzt keine Ahnung was da sinnvoll ist
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)
    
    features = state_to_features(game_state)

    action = ACTIONS[np.argmax(model_evaluation(self.model, features))]
    #action = rule_based(features)
    #action = act_(self, game_state)
    return action

def state_to_features(game_state: dict):
    start_time = time.time()
    features = []

    _, _, bomb_available, agent_pos = game_state['self']
    field = game_state['field']
    coins = game_state['coins']
    pos_x, pos_y = agent_pos

    #print(0, time.time()-start_time)
    # direction to nearest safe spot
    goal = lambda x, y: field_is_safe(game_state, x, y)
    features.append(shortest_path(game_state, field, pos_x, pos_y, goal))

    #print(1, time.time()-start_time)
    # direction to nearest coin
    goal = lambda x, y: (x, y) in coins
    features.append(shortest_path(game_state, field, pos_x, pos_y, goal, safe_path=True))

    #print(2, time.time()-start_time)
    # direction to nearest crate
    goal = lambda x, y: (field[x, y+1] == 1 or
                                field[x, y-1] == 1 or
                                field[x+1, y] == 1 or
                                field[x-1, y] == 1)
    features.append(shortest_path(game_state, field, pos_x, pos_y, goal, safe_path=True))

    #print(3, time.time()-start_time)
    # safe to bomb 
    goal = lambda x, y: field_is_safe(game_state, x, y, pos_x, pos_y)
    no_way_out = shortest_path(game_state, field, pos_x, pos_y, goal) == 0
    features.append(0 if no_way_out or not bomb_available else 1)

    #print(4, time.time()-start_time)
    return features

def field_is_safe(game_state, pos_x, pos_y, bomb_x=None, bomb_y=None):
    '''
    check if the given field is safe, ie: 
    there is no explosion and no explosion in the near future happening on this field
    '''
    field = game_state['field']
    bombs = game_state['bombs'].copy()
    if bomb_x and bomb_y:
        bombs.append(((bomb_x, bomb_y), 3))
    explosion_map = game_state['explosion_map']
    safe = True

    if explosion_map[pos_x, pos_y] != 0:
        safe = False

    for (x, y), t in bombs:
        if (pos_x == x and abs(y - pos_y) <= 3):
            s = 1 if y > pos_y else -1 
            wall = False
            for d in range(s, y-pos_y, s):
                if field[x, pos_y+d] == -1:
                    wall = True
            if not wall:
                safe = False

        if (pos_y == y and abs(x - pos_x) <= 3):
            s = 1 if x > pos_x else -1 
            wall = False
            for d in range(s, x-pos_x, s):
                if field[pos_x+d, y] == -1:
                    wall = True
            if not wall:
                safe = False

    return safe

def point_in_list(x, y, l):
    if len(l) == 0: return False
    return np.min(np.sum(abs(np.array(l)[:, :2] - [x, y]), axis=1)) == 0

def shortest_path(game_state, field, x_s, y_s, goal, safe_path=False):
    fields_visited = []
    fields_to_check = deque([[x_s, y_s, None]])
    while fields_to_check:
        x, y, i = fields_to_check.popleft()
        
        if goal(x, y):
            i_current = i
            while True:
                if x == x_s and y == y_s:
                    return 1
                if x == x_s and y == y_s+1:
                    return 2
                if x == x_s and y == y_s-1:
                    return 3
                if x == x_s+1 and y == y_s:
                    return 4
                if x == x_s-1 and y == y_s:
                    return 5
                x, y, i_current = fields_visited[i_current]

        fields_visited.append([x, y, i])
        i = len(fields_visited) - 1
        
        safe = not safe_path or field_is_safe(game_state, x-1, y)
        if field[x-1, y] == 0 and not point_in_list(x-1, y, fields_visited) and not point_in_list(x-1, y, fields_to_check) and safe:
            fields_to_check.append([x-1, y, i])
        safe = not safe_path or field_is_safe(game_state, x+1, y)
        if field[x+1, y] == 0 and not point_in_list(x+1, y, fields_visited) and not point_in_list(x+1, y, fields_to_check) and safe:
            fields_to_check.append([x+1, y, i])
        safe = not safe_path or field_is_safe(game_state, x, y-1)
        if field[x, y-1] == 0 and not point_in_list(x, y-1, fields_visited) and not point_in_list(x, y-1, fields_to_check) and safe:
            fields_to_check.append([x, y-1, i])
        safe = not safe_path or field_is_safe(game_state, x, y+1)
        if field[x, y+1] == 0 and not point_in_list(x, y+1, fields_visited) and not point_in_list(x, y+1, fields_to_check) and safe:
            fields_to_check.append([x, y+1, i])

    return 0


def rule_based(features):
    outputs = ['NONE', 'CURRENT', 'DOWN', 'UP', 'RIGHT', 'LEFT']
    safe_direction = outputs[features[0]]
    coin_direction = outputs[features[1]]
    crate_direction = outputs[features[2]]
    bad_bomb = features[3]

    if safe_direction in ['DOWN', 'UP', 'RIGHT', 'LEFT']:
        return safe_direction

    if coin_direction in ['DOWN', 'UP', 'RIGHT', 'LEFT']:
        return coin_direction

    if crate_direction in ['DOWN', 'UP', 'RIGHT', 'LEFT']:
        return crate_direction

    if crate_direction == 'CURRENT' and bad_bomb == 1:
        return 'BOMB'

    return 'WAIT'
    