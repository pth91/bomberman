import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # Own events
    old_f = state_to_features(old_game_state)
    if type(old_f) == type(state_to_features(new_game_state)):
        action_num = np.argmax(self_action == np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])[:])
        if old_f[1] != 5 and old_f[1] != action_num:
            events.append('NOT_DOGED') # ignored doge direction while in danger
        if self_action == best_action(old_f):
            events.append('BEST')
        else:
            events.append('NOT_BEST')
        self.logger.debug(f'{best_action(old_f)}')
            
# war nur debug zeugs
#       _,_,_, o = old_game_state['self']
#       _,_,_, n = new_game_state['self']
#       x = o[0] - n[0]
#       y = o[1] - n[1]
 #       self.logger.debug(f'{self_action} is {x} and y {y} and rec {state_to_features(old_game_state)[1]}')

    # state_to_features is defined in callbacks.py
    
    # Update sobald deque voll ist oder wenn bombe gesetzt IDEE: wurde damit die konsequenzen der Bombe noch in das bomben legen mit hinein gerechnet werden. Das beschleunigt wenn überhaupt nur das lernen am anfang und kann gerne weg
    if len(self.transitions) == 4 or self_action == 'BOMB':
        q_update_transitions(self)
        
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))) # war schon so definiert hab ich nicht dran geändert
    
    # im ersten Step is old_state= None das kann q_update nicht hendeln deshalb schmeißt man es raus
    if type(state_to_features(old_game_state)) != type(state_to_features(new_game_state)):
        self.logger.debug(f'Encountered game type problem in step {new_game_state["step"]}')
        self.transitions.pop()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, state_to_features(last_game_state), reward_from_events(self, events))) # hab None duch einen 2ten old_state ersetzt da q_update type None nicht hendeln kann

    q_update_transitions(self)
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    # Das folgende ist passt zumindes für coins ohne crates
    game_rewards2 = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.MOVED_LEFT: -.01, #IDEA: kein trödeln
        e.MOVED_RIGHT: -.01,
        e.MOVED_UP: -.01,
        e.MOVED_DOWN: -.01,
        e.WAITED: -.01,
        e.INVALID_ACTION: -1, 
        e.BOMB_DROPPED: -0.1, #IDEA: less bombs
        e.KILLED_SELF: -4,
        e.GOT_KILLED: -4.5, #IDEA: KILLED_SELF is better cause no opponent gets points
        e.SURVIVED_ROUND: .5, #IDEA: Good, but no coward
     #   e.CRATE_DESTROYED: 0.1, #IDEA: for training step 2.
        'NOT_DOGED': -1, #IDEA: if in danger doge!
    }
    game_rewards = {'BEST': 1, 'NOT_BEST': -1, e.INVALID_ACTION: -1}
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# q-learing Update
def q_update_transitions(self):
    alpha = 0.1 # learning rate, keine Ahnung hab sie random gesetzt und bisher nicht geändert
    gamma = 0.5 #importace of future
    
    reward = 0
    for i in range(len(self.transitions)):
            old, a, new, rewardn = self.transitions.pop()
            action_num = np.argmax(a == np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])[:])
            reward = rewardn + reward *0  #ow(0.5, i)
            # da das Feld gedreht und gespiegelt auch ein valid state ist
            # update man alle 8 states, wurde in der Vorlsung empfohlen und hilf gegeb seltsames verhalten an einzelnen Randfeldern
            for j in range(1):
                self.model[0, old[0], old[1], old[2], old[3], old[4], old[5], old[6], action_num] = (1 - alpha) * self.model[0, old[0], old[1], old[2], old[3], old[4], old[5], old[6], action_num] + alpha * (reward + gamma * np.max(self.model[0, new[0], new[1], new[2], new[3], new[4], new[5], new[6]]))
                self.model[1, old[0], old[1], old[2], old[3], old[4], old[5], old[6], action_num] += 1
             #  old, new, action_num = rotate_features(old, new, action_num)
                if j == 3:
                    old, new, action_num = mirrow_features(old, new, action_num)
            
#rotate
def rotate_features(old, new, action):
    old = old[[0, 1, 4, 5, 3, 2, 6]]
    new = new[[0, 1, 4, 5, 3, 2, 6]]
    old[1] = [1, 2, 3, 0, 4, 5][old[1]]
    new[1] = [1, 2, 3, 0, 4, 5][new[1]]
    old[6] = [1, 2, 3, 0][old[6]]
    new[6] = [1, 2, 3, 0][new[6]]
    action = [1, 2, 3, 0, 4, 5][action] 
    return old, new, action

def mirrow_features(old, new, action):
    old = old[[0, 1, 3, 2, 4, 5, 6]]
    new = new[[0, 1, 3, 2, 4, 5, 6]]
    old[1] = [0, 3, 2, 1, 4, 5][old[1]]
    new[1] = [0, 3, 2, 1, 4, 5][new[1]]
    old[6] = [0, 3, 2, 1][old[6]]
    new[6] = [0, 3, 2, 1][new[6]]
    action = [0, 3, 2, 1, 4, 5][action]
    return old, new, action

def best_action(features):
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    F2A = [0, 0, 3, 1, 0, 2]
    aim = [4, 3, 5, 2, 4, 3, 5, 2, 6][features[6]]
    if features[1] <= 3: #dog if in danger highest priority
        return ACTIONS[features[1]]
    if features[0] == 1:
        return ACTIONS[5]
    if features[6] <= 3 and features[aim] == 0:#go to t if one is reachable
        return ACTIONS[features[6]]
    if np.sum(features[2:6] == -1) == 1: #if possible walk to the inner part of the board
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
    best_action = np.argmax(features[directions[:]] == 0)
    if features[directions[best_action]] != 0: # no direction is free to walk
        return ACTIONS[4]
    return ACTIONS[F2A[directions[best_action]]]   