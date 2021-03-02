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
    if type(state_to_features(old_game_state)) == type(state_to_features(new_game_state)):
        action_num = np.argmax(self_action == np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])[:])
        if state_to_features(old_game_state)[1] != 5 and state_to_features(old_game_state)[1] != action_num:
            events.append('NOT_DOGED') # ignored doge direction while in danger
        if state_to_features(old_game_state)[6] <= 3 and state_to_features(old_game_state)[1] == action_num:
            events.append('TO_COIN') # moved to coin
        if state_to_features(old_game_state)[6] <= 3 and state_to_features(old_game_state)[1] != action_num:
            events.append('NOT_TO_COIN') #moved not to coin while a coin is reachable
        if state_to_features(old_game_state)[0] == 3 and 5 == action_num:
            events.append('SUIZIDE') #bomb drop at impossible to doge position
        if state_to_features(old_game_state)[6] <= 3 and 5 == action_num:
            events.append('UNNESSACARY') #bomb eventhure a coin is reachable
            # das war noch weil der agent grundlos bomben gespammt hat kann/muss eventuell weg wenn gegner ins Spiel kommen
            
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
    game_rewards = {
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
        e.CRATE_DESTROYED: 0.1, #IDEA: for training step 2.
        'NOT_DOGED': -1, #IDEA: if in danger doge!
        'TO_COIN': 0.1, #IDEA: NOT_TO_COIN+TO_COIN=0 no exploitation
        'NOT_TO_COIN': -0.1,
        'UNNESSACARY': -1, #Note: muss eventuell weg war um bomben legen während traing step 1. zu verhindern
        'SUIZIDE': -4
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# q-learing Update
def q_update_transitions(self):
    alpha = 0.1 # learning rate, keine Ahnung hab sie random gesetzt und bisher nicht geändert
    
    for i in range(len(self.transitions)):
            old, a, new, reward = self.transitions.pop()
            action_num = np.argmax(a == np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])[:])
            # da das Feld gedreht und gespiegelt auch ein valid state ist
            # update man alle 8 states, wurde in der Vorlsung empfohlen und hilf gegeb seltsames verhalten an einzelnen Randfeldern
            for j in range(8):
                self.model[old[0], old[1], old[2], old[3], old[4], old[5], old[6], old[7], action_num] = (1 - alpha) * self.model[old[0], old[1], old[2], old[3], old[4], old[5], old[6], old[7], action_num] + alpha * (reward + np.max(self.model[new[0], new[1], new[2], new[3], new[4], new[5], new[6], new[7]]))
                old, new, action_num = rotate_features(old, new, action_num)
                if j == 3:
                    old, new, action_num = mirrow_features(old, new, action_num)
            
#rotate
def rotate_features(old, new, action):
    old = old[[0, 1, 5, 4, 2, 3, 6, 7]]
    new = new[[0, 1, 5, 4, 2, 3, 6, 7]]
    #wegen 4= no x und 5= not reachable etc konnte ich es nicht modulo machen
    if old[1]<=3:
        old[1] = (old[1] + 1) % 4
    if old[6]<=3:
        old[6] = (old[6] + 1) % 4
    if old[7]<=3:
        old[7] = (old[7] + 1) % 4
    if new[1]<=3:
        new[1] = (new[1] + 1) % 4
    if new[6]<=3:
        new[6] = (new[6] + 1) % 4
    if new[7]<=3:
        new[7] = (new[7] + 1) % 4
    if action<=3:
        action = (action + 1) % 4
    return old, new, action

def mirrow_features(old, new, action):
    old = old[[0, 1, 3, 2, 4, 5, 6, 7]]
    new = new[[0, 1, 3, 2, 4, 5, 6, 7]]
    if old[1] == 1 or old[1] == 3:
        old[1] = (old[1] + 2) % 4
    if old[6] == 1 or old[6] == 3:
        old[6] = (old[6] + 2) % 4
    if old[7] == 1 or old[7] == 3:
        old[7] = (old[7] + 2) % 4
    if new[1] == 1 or new[1] == 3:
        new[1] = (new[1] + 2) % 4
    if new[6] == 1 or new[6] == 3:
        new[6] = (new[6] + 2) % 4
    if new[7] == 1 or new[7] == 3:
        new[7] = (new[1] + 2) % 4
    if action == 1 or action == 3:
        action = (action + 2) % 4
    return old, new, action