import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict
from enum import Enum
from copy import deepcopy

"""
Grid World problem
"""

class Action(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

class Environment:
    def __init__(self, n):
        self.world_size = (n, n)
        self.terminal_states = [(0,0), (n-1, n-1)]
    
    def is_terminal(self, state : Tuple[int]):
        return state in self.terminal_states

    def get_all_states(self):
        states = []
        for i in range(self.world_size[0]):
            for j in range(self.world_size[1]):
                states.append(
                    (i,j)
                )
        return states
    
    def get_reward(self, state : Tuple[int], action : int, new_state : Tuple[int]): # r(s,a,s')
        if self.is_terminal(new_state):
            return 0.0
        return -1.0
    
    def dynamics(self, state : Tuple[int], action : int): # p(s',r|s,a)
        x, y = state
        if action == Action.LEFT:
            x = max(x - 1, 0)
        elif action == Action.RIGHT:
            x = min(x + 1, self.world_size[0] - 1)
        elif action == Action.UP:
            y = min(y + 1, self.world_size[1] - 1)
        elif action == Action.DOWN:
            y = max(y - 1, 0)
        new_state = (x, y)
        reward = self.get_reward(state, action, new_state)
        return {
            new_state : {
                reward : 1.0 # the probability is 1.0 because we make it deterministic
            }
        }

class Agent:
    def __init__(self, env : Environment, discount_factor : int=0.5):
        self.state_values = self.init_state_values(env)
        self.state_action_values = self.init_state_action_values(env)
        self.current_state = (
            np.random.randint(low=0, high=env.world_size[0]),
            np.random.randint(low=0, high=env.world_size[1])
        )
        self.discount_factor = discount_factor
    
    def init_state_values(self, env : Environment): # V(s)
        state_values = np.random.rand(*env.world_size)
        for terminal_state in env.terminal_states:
            state_values[terminal_state[0], terminal_state[1]] = 0.0
        return state_values

    def available_actions(self, state : Tuple[int], env : Environment):
        actions = [a for a in Action]

        if state[0] == 0:
            actions.remove(Action.LEFT)
        elif state[0] == env.world_size[0]-1:
            actions.remove(Action.RIGHT)
        
        if state[1] == 0:
            actions.remove(Action.DOWN)
        elif state[1] == env.world_size[1]-1:
            actions.remove(Action.UP)
        
        return actions

    def init_state_action_values(self, env : Environment): # q_π(s,a)
        state_action_values = {}
        states = env.get_all_states()
        for s in states:
            available_actions = self.available_actions(s, env)
            state_action_values[s] = {
                a : 0.0 for a in available_actions
            }
        return state_action_values
    
    def policy(self, state : Tuple[int]): # π(a|s)
        action_values  = self.state_action_values[state]
        return np.random.choice(list(action_values.keys()))
    
    def generate_episode(self, env : Environment, start_state : Tuple[int, int], max_iter : int=1000):
        current_state = start_state # S0
        current_action = self.policy(current_state) # A0
        episode = []
        for _ in range(max_iter):
            dynamics = env.dynamics(current_state, current_action)
            all_dynamics = []
            all_probs = []
            for new_state, rewards in dynamics.items():
                for reward, p in rewards.items():
                    all_dynamics.append(
                        (new_state, reward)
                    )
                    all_probs.append(p)
            chosen_dynamics_index = np.random.choice([i for i in range(len(all_dynamics))], p=all_probs)
            new_state, reward = all_dynamics[chosen_dynamics_index]
            episode.append(
                (current_state, current_action, reward) # St-1, At-1, Rt
            )
            current_state = new_state
            if env.is_terminal(current_state):
                break
            current_action = self.policy(current_state)
        return episode
    
    def first_visit_mc_prediction(self, env : Environment, iterations : int=10000): # we set initial state to be in the middle
        state_values = self.state_values.copy()
        all_states = env.get_all_states()
        returns = {
            s : [] for s in all_states
        }

        initial_state_idx = np.random.choice(
            [i for i in range(len(all_states)) if not env.is_terminal(all_states[i])]
        )
        initial_state = all_states[initial_state_idx]

        for _ in tqdm(range(iterations)):
            episode = self.generate_episode(env, initial_state)
            states = [el[0] for el in episode]
            g = 0
            for i in range(len(episode)):
                s_t, a_t, r_t_plus_1 = episode[len(episode)-i-1]
                g = self.discount_factor * g + r_t_plus_1
                if s_t not in states[:len(episode)-i-1]:
                    returns[s_t].append(g)
                    state_values[s_t[0], s_t[1]] = np.mean(returns[s_t])
        
        self.state_values = state_values

if __name__ == "__main__":
    N = 5
    DISCOUNT_FACTOR = 0.5
    env = Environment(N)
    agent = Agent(env, DISCOUNT_FACTOR)

    print("Terminal states:")
    print(env.terminal_states)

    print("Initial state values:")
    print(agent.state_values)

    print("First visit MC prediction...")
    agent.first_visit_mc_prediction(env)

    print("Result state values:")
    print(agent.state_values)