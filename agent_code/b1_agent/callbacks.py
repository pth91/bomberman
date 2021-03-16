import os
import pickle
import random

import numpy as np
from sklearn import tree

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
#   if self.train or not os.path.isfile("my-saved-model.pt"):
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
      #  weights = np.random.rand(len(ACTIONS))
       # self.model = weights / weights.sum()
        self.model = np.zeros((2, 2, 6, 2, 2, 2, 2, 4, 6)) # 1-7 dim featurespace, 8 #Actions, 0 switch model/count
        
    elif self.train:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
#       for a in range(2):
#           for b in range(6):
#               for c in range(2):
#                   for d in range(2):
#                       for e in range(2):
#                           for f in range(2):
#                               for g in range(4):
#                                   print(a,b,c,d,e,f,g, self.model[a,b,c,d,e,f,g])
        print(self.model[1])
        self.clf = build_regression(self)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = 0 #Hab ich gesetzt keine Ahnung was da sinnvoll ist
    tol = 0.2 #if action1 - action2 <tol => regression
    minimal_tries = 3 #minimal data on action to trust
    
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # Hab die bomb und wait rate runter fürs coin collecting training
        return np.random.choice(ACTIONS, p=[.2, .2, .29, .29, .01, .01])
    
    if self.train and random.random() < 1:
        return best_action(self, state_to_features(game_state))
    
    if hasattr(self, 'model'):
        features = state_to_features(game_state)
        if type(features) != type(np.zeros(2)):
            self.logger.debug(f'type error {type(features)} in step {game_state["step"]}')
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        action = np.argmax(self.model[0, features[0], features[1] ,features[2], features[3], features[4], features[5], features[6]])
        count = self.model[1, features[0], features[1] ,features[2], features[3], features[4], features[5], features[6], action]
        action2 = np.argmax(self.model[0, features[0], features[1] ,features[2], features[3], features[4], features[5], features[6]] - np.eye(6)[action] * tol)
        if action != action2 or count < minimal_tries:
            return regression(features, self.clf)
        return ACTIONS[action]
 #   self.logger.debug("Querying model for action.") 
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    _, _, bomb_ready, a_pos = game_state['self']
    field = game_state['field'] + np.floor(game_state['explosion_map']* 0.5)* 5  # setzt explosions mit timer 2 auf 5, exp mit timer 1 sind irrelevant
    direction_opponent = 5
    
    # Setzt die anderen Agents aufs Feld
    for foe in game_state['others']:
        _, _, _, pos = foe
        x, y = pos
        field[x, y] = 3
        if x == a_pos[0] and abs(y - a_pos[1]) <= 3:
            direction_opponent = 4
        if y == a_pos[1] and abs(x- a_pos[0]) <= 3:
            direction_opponent = 4
        
    # Setzt sichtbare coins aufs Feld
    for x, y in game_state['coins']:
        field[x, y] = 2
    
    #Setzt bomben=4 und 5 für Felder die in 1 Schritt hochgehen, ..., 8 für in 4 Schritten hochgehen
    for bomb in game_state['bombs']:
        pos, t = bomb 
        x, y = pos
        field[x, y] = 4
        free_direction = np.array([1, 1, 1, 1])
        for i in range(1, 4): # Setzt die Felder die von der explosion der bombe erreichtwerden auf 5,..,8 entsprechen dem timer der bombe
            if free_direction[0] == 1:
                if field[x-i, y] != -1:
                    if field[x-i, y] == 0 or field[x-i, y]> t + 4:
                        field[x-i, y] = t + 5
                else:
                    free_direction[0] = 0
            if free_direction[1] == 1:
                if field[x+i, y] != -1:
                    if field[x+i, y] == 0 or field[x+i, y]> t + 4:
                        field[x+i, y] = t + 5
                else:
                    free_direction[1] = 0
            if free_direction[2] == 1:
                if field[x, i+y] != -1:
                    if field[x, i+y] == 0 or field[x, i+y]> t+ 4:
                        field[x, i+y] = t + 5
                else:
                    free_direction[2] = 0
            if free_direction[3] == 1:
                if field[x, y-i] != -1:
                    if field[x, y-i] == 0 or field[x, y-i]> t+ 4:
                        field[x, y-i] = t + 5
                else:
                    free_direction[3] = 0
    
    #search coin
    if len(game_state['coins'])>0:
        board = np.minimum(field, 0) - (field == 1) - (field == 3) - (field == 4)
        board[a_pos[0], a_pos[1]] = - 1
        paths = []
        direction_coin = 5 # default 5 bedeutet keine coin erreichbar
        # Erster Schritt außerhalb der while um Startrichtung zu setzten
        # 0-3 entsprechent der Reihenfolge in ACTION für custom events in train.py
        if board[a_pos[0] + 1, a_pos[1]] == 0 and field[a_pos[0] + 1, a_pos[1]] != 5:
            paths.append((a_pos[0] + 1, a_pos[1], 1)) #1 Pfad begint mit RIGHT
            board[a_pos[0] + 1, a_pos[1]] = -1 # Damit eine Felder wiederholt werden
        if board[a_pos[0] - 1, a_pos[1]] == 0 and field[a_pos[0] - 1, a_pos[1]] != 5:
            paths.append((a_pos[0] - 1, a_pos[1], 3)) #3 Pfad beginnt LEFT
            board[a_pos[0] - 1, a_pos[1]] = -1
        if board[a_pos[0], a_pos[1]+ 1] == 0 and field[a_pos[0], 1 + a_pos[1]] != 5:
            paths.append((a_pos[0], a_pos[1]+ 1, 2)) #2 Pfad beginnt DOWN
            board[a_pos[0], a_pos[1]+ 1] = -1
        if board[a_pos[0], a_pos[1]- 1] == 0 and field[a_pos[0], a_pos[1]- 1] != 5:
            paths.append((a_pos[0], a_pos[1]- 1, 0)) #0 Pfad beginnt UP
            board[a_pos[0], a_pos[1]- 1] = -1
        while len(paths)>0 and direction_coin == 5: #Breitensuche
            x, y , z = paths.pop(0)
            if field[x, y] == 2:
                direction_coin = z
            if board[x + 1, y] == 0:
                paths.append((x + 1, y, z))
                board[x + 1, y] = -1
            if board[x - 1, y] == 0:
                paths.append((x - 1, y, z))
                board[x - 1, y] = -1
            if board[x, y + 1] == 0:
                paths.append((x, y + 1, z))
                board[x, y + 1] = -1
            if board[x, y - 1] == 0:
                paths.append((x, y - 1, z))
                board[x, y - 1] = -1
    else:
        direction_coin = 8 # 4 = es existieren keine sichtbaren Münzen auf dem Feld
    #search next crate if no coin reachable
    if direction_coin == 8: #and np.sum(field[:, :] == 1) >= 1
        board = np.minimum(field, 0)  - (field == 3) - (field == 4)
        board[a_pos[0], a_pos[1]] = - 1
        paths = []
        # Erster Schritt außerhalb der while um Startrichtung zu setzten
        # 0-3 entsprechent der Reihenfolge in ACTION für custom events in train.py
        if board[a_pos[0] + 1, a_pos[1]] == 0 and field[a_pos[0] + 1, a_pos[1]] != 5:
            paths.append((a_pos[0] + 1, a_pos[1], 5)) #1 Pfad begint mit RIGHT
            board[a_pos[0] + 1, a_pos[1]] = -1 # Damit eine Felder wiederholt werden
        if board[a_pos[0] - 1, a_pos[1]] == 0 and field[a_pos[0] - 1, a_pos[1]] != 5:
            paths.append((a_pos[0] - 1, a_pos[1], 7)) #3 Pfad beginnt LEFT
            board[a_pos[0] - 1, a_pos[1]] = -1
        if board[a_pos[0], a_pos[1]+ 1] == 0 and field[a_pos[0], a_pos[1]+ 1] != 5:
            paths.append((a_pos[0], a_pos[1]+ 1, 6)) #2 Pfad beginnt DOWN
            board[a_pos[0], a_pos[1]+ 1] = -1
        if board[a_pos[0], a_pos[1]- 1] == 0 and field[a_pos[0], a_pos[1]- 1] != 5:
            paths.append((a_pos[0], a_pos[1]- 1, 4)) #0 Pfad beginnt UP
            board[a_pos[0], a_pos[1]- 1] = -1
        while len(paths)>0 and direction_coin == 8: #Breitensuche
            x, y , z = paths.pop(0)
            if field[x, y] == 1:
                direction_coin = z
            if board[x + 1, y] == 0:
                paths.append((x + 1, y, z))
                board[x + 1, y] = -1
            if board[x - 1, y] == 0:
                paths.append((x - 1, y, z))
                board[x - 1, y] = -1
            if board[x, y + 1] == 0:
                paths.append((x, y + 1, z))
                board[x, y + 1] = -1
            if board[x, y - 1] == 0:
                paths.append((x, y - 1, z))
                board[x, y - 1] = -1
    
    #search opponent
    # Selbe wie für coins da agents in coin search als wall gehandthabt werden hab ich ne zweite Suche gemacht würde sich sichher aber zusammen in eine Breitensuche machen lassen
    
    if len(game_state['others'])>0 and direction_opponent == 5:
        board = np.minimum(field, 0) - (field == 1) - (field == 4)
        board[a_pos[0], a_pos[1]] = - 1
        paths = []
        direction_opponent = 5
        if board[a_pos[0] + 1, a_pos[1]] == 0 and field[a_pos[0] + 1, a_pos[1]] != 5:
            paths.append((a_pos[0] + 1, a_pos[1], 1))
            board[a_pos[0] + 1, a_pos[1]] = -1
        if board[a_pos[0] - 1, a_pos[1]] == 0 and field[a_pos[0] - 1, a_pos[1]] != 5:
            paths.append((a_pos[0] - 1, a_pos[1], 3))
            board[a_pos[0] - 1, a_pos[1]] = -1
        if board[a_pos[0], a_pos[1]+ 1] == 0 and field[a_pos[0], a_pos[1]+ 1] != 5:
            paths.append((a_pos[0], a_pos[1]+ 1, 2))
            board[a_pos[0], a_pos[1]+ 1] = -1
        if board[a_pos[0], a_pos[1]- 1] == 0 and field[a_pos[0], a_pos[1]- 1] != 5:
            paths.append((a_pos[0], a_pos[1]- 1, 0))
            board[a_pos[0], a_pos[1]- 1] = -1
        while len(paths)>0 and direction_opponent == 5:
            x, y , z = paths.pop(0)
            if field[x, y] == 3:
                direction_opponent = z
            if board[x + 1, y] == 0:
                paths.append((x + 1, y, z))
                board[x + 1, y] = -1
            if board[x - 1, y] == 0:
                paths.append((x - 1, y, z))
                board[x - 1, y] = -1
            if board[x, y + 1] == 0:
                paths.append((x, y + 1, z))
                board[x, y + 1] = -1
            if board[x, y - 1] == 0:
                paths.append((x, y - 1, z))
                board[x, y - 1] = -1
    
    #doge explosion
    doge_dir = find_doge_direction(field, a_pos) # findet 0-3 richtung um drohender expl auszuweichen, 4 exp ist unausweichbar, 5 es existiert keine Gefahr auf dem aktuellen Feld
    
    #is a bomb drop safe?
    if bomb_ready == 1: #1=bomb ready
        field_b = np.copy(field)
        x, y = a_pos
        t = 4
        field_b[x, y] = 4 # Wenn man auf aktuelles Feld bomb legen würde...
        field_b[field_b >= 6] -= 1
        free_direction = np.array([1, 1, 1, 1])
        #Eventuell zu ner function machen
        for i in range(1, 4): #Plaziert wie oben bei bomb 
            if free_direction[0] == 1:
                if field[x-i, y] != -1:
                    if field[x-i, y] == 0 or field[x-i, y]> t + 4:
                        field_b[x-i, y] = t + 4
                else:
                    free_direction[0] = 0
            if free_direction[1] == 1:
                if field[x+i, y] != -1:
                    if field[x+i, y] == 0 or field[x+i, y]> t + 4:
                        field_b[x+i, y] = t + 4
                else:
                    free_direction[1] = 0
            if free_direction[2] == 1:
                if field[x, i+y] != -1:
                    if field[x, i+y] == 0 or field[x, i+y]> t+ 4:
                        field_b[x, i+y] = t + 4
                else:
                    free_direction[2] = 0
            if free_direction[3] == 1:
                if field[x, y-i] != -1:
                    if field[x, y-i] == 0 or field[x, y-i]> t+ 4:
                        field_b[x, y-i] = t + 4
                else:
                    free_direction[3] = 0
        if find_doge_direction(field_b, a_pos) == 4: #4=unausweichbar
            bomb_ready = 2 #2 = bombe legen ist SD
    
    # For example, you could construct several channels of equal shape, ...
    channels = []
    
    bomb_d = 0
    sumcrat = (field[a_pos[0]- 1, a_pos[1]] == 1) + (field[a_pos[0]+ 1, a_pos[1]] == 1) + (field[a_pos[0], a_pos[1]- 1] == 1) + (field[a_pos[0], a_pos[1]+ 1] == 1) 
    if int(bomb_ready) == 1 and sumcrat >= 1 and direction_coin>=4:
        bomb_d = 1
    if bomb_ready == 1 and direction_opponent == 4:
        bomb_d = 1
    channels.append(bomb_d) #bomb 1, not 0
#    channels.append(field[a_pos[0], a_pos[1]]) #current
    channels.append(doge_dir)  #0-3 direction to doge exp, 4 undogeable, 5 no need to doge
    # die anliegenden Felder mit: -1 wall, 0 free, 1 crates, 2 coin, 3 agent,  4 bomb, 5 explosuion now/ explosion in 1, 6 exp in 2, 7 exp in 3, 8 exp in 4
    reduce = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1]) 
    channels.append(reduce[int(field[a_pos[0]- 1, a_pos[1]])]) #left
    channels.append(reduce[int(field[a_pos[0]+ 1, a_pos[1]])])# right
    channels.append(reduce[int(field[a_pos[0], a_pos[1]- 1])]) #up
    channels.append(reduce[int(field[a_pos[0], a_pos[1]+ 1])]) #down

   #channels.append(dist_coin) würde die feature space zu groß machen eventuell mit buckets? 0 nah, 1 mittel, 2 fern?
    #0-3 direction of nearest coin, 4-7 direction of nearest crate if no coin, 8 no coin or crate
    direction_target = direction_coin%4
    if direction_coin == 8:
        direction_target = direction_opponent%4
    channels.append(direction_target)
    #0-3 direction of nearest opp, 4 opp in bomb range, 5 no opp reachabel/exist
  # channels.append(dist_opponent)
 #   channels.append(direction_opponent)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    stacked_channels.astype(int)
    # and return them as a vector
    
    #Overall feature  0      1     2    3     4     5     6           
    #dimension        2      6     2    2     2     2     4        
    #               bomb? doge_d  <-   ->    up    down target_d
    return stacked_channels.reshape(-1).astype(int)

def find_doge_direction(field: np.array, a_pos: np.array) -> int:
    if field[a_pos[0], a_pos[1]] != 0:
        board = np.minimum(field, 0) - (field == 1) - (field == 3) - (field == 4)
 #      board[a_pos[0], a_pos[1]] = - 1
        paths = []
        doge_dir = 4
        if board[a_pos[0] + 1, a_pos[1]] == 0 and field[a_pos[0] + 1, a_pos[1]] != 5:
            paths.append((a_pos[0] + 1, a_pos[1], 1, 1))
        if board[a_pos[0] - 1, a_pos[1]] == 0 and field[a_pos[0] - 1, a_pos[1]] != 5:
            paths.append((a_pos[0] - 1, a_pos[1], 3, 1))
        if board[a_pos[0], a_pos[1]+ 1] == 0 and field[a_pos[0],  1+ a_pos[1]] != 5:
            paths.append((a_pos[0], a_pos[1]+ 1, 2, 1))
        if board[a_pos[0], a_pos[1]- 1] == 0 and field[a_pos[0], a_pos[1]- 1] != 5:
            paths.append((a_pos[0], a_pos[1]- 1, 0, 1))
        time = 1
        while doge_dir == 4 and len(paths)>0:
            x, y , z, time = paths.pop(0)
            if field[x, y] - time <= 3 and time <=4:
                doge_dir = z
            if board[x + 1, y] == 0 and field[x + 1, y] - time != 5 and field[x + 1, y] - time != 4:
                paths.append((x + 1, y, z, time + 1))
            if board[x - 1, y] == 0 and field[x - 1, y] - time != 5 and field[x - 1, y] - time != 4:
                paths.append((x - 1, y, z, time + 1))
            if board[x, y + 1] == 0 and field[x, y + 1] - time != 5 and field[x, y + 1] - time != 4:
                paths.append((x, y + 1, z, time + 1))
            if board[x, y - 1] == 0 and field[x, y - 1] - time != 5 and field[x, y - 1] - time != 4:
                paths.append((x, y - 1, z, time + 1))
    else:
        doge_dir = 5
    return doge_dir

def best_action(self, features):
    self.logger.debug("Choosing best a")
    F2A = [0, 0, 3, 1, 0, 2]
    aim = [4, 3, 5, 2, 4, 3, 5, 2, 6][features[6]]
    if features[1] <= 3: #dog if in danger highest priority
        return ACTIONS[features[1]]
    if features[0] == 1: #bomb if t makes sens
        return ACTIONS[5]
    if features[6] <= 3 and features[aim] == 0:#go to target if one is reachable
        return ACTIONS[features[6]]
    if np.sum(features[2:6] == -1) == 1: #if possible walk to the inner part of the board
        self.logger.debug("Rand action")
        if features[2] == -1 and features[3] == 0:
            return ACTIONS[1]
        if features[3] == -1 and features[2] == 0:
            return ACTIONS[3]
        if features[4] == -1 and features[5] == 0:
            return ACTIONS[2]
        if features[5] == -1 and features[4] == 0:
            return ACTIONS[0]
    directions = np.array([2, 3, 4, 5])
    np.random.shuffle(directions)
    self.logger.debug(f'f6: {features[6]}')
    best_action = np.argmax(features[directions[:]] == 0)
    if features[directions[best_action]] != 0: # no direction is free to walk
        self.logger.debug(f'feature 1: {features[1]}')
        return ACTIONS[4]
    self.logger.debug(f'action: {ACTIONS[F2A[directions[best_action]]]}')
    return ACTIONS[F2A[directions[best_action]]]

def build_regression(self):
    dims = (self.model[0]).shape
    channel = []
    y = []
    for a in range(dims[0]):
        for b in range(dims[1]):
            for c in range(dims[2]):
                for d in range(dims[3]):
                    for e in range(dims[4]):
                        for f in range(dims[5]):
                            for g in range(dims[6]):
                                action = np.argmax(self.model[0, a, b, c, d, e, f, g])
                                if self.model[1, a, b, c, d, e, f, g, action]  >= 3:
                                    channel.append([a, b, c, d, e, f, g])
                                    y.append([action])
                                
    
    X = np.stack(channel)
    y = np.stack(y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    return clf

def regression(features, clf):
    print(clf.predict([features]))
    return ACTIONS[clf.predict([features])[0]]
