import numpy as np
from tqdm import tqdm
from typing import Tuple
from enum import Enum

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
    
    def policy_prob(self, state : Tuple[int]): # π(a|s)
        action_values  = self.state_action_values[state]
        exp_action_values = {a : np.exp(v) for a, v in action_values.items()}
        action_prob = {a : e/sum(exp_action_values.values()) for a, e in exp_action_values.items()} # we want to use softmax
        return action_prob
    
    def policy(self, state : Tuple[int]): # π(s) deterministic
        action_prob = self.policy_prob(state)
        chosen_action = max(action_prob, key=action_prob.get)
        return chosen_action
    
    def sweep(self, env : Environment, state_values : np.ndarray):
        states = env.get_all_states()
        delta = 0.0
        for state in states:
            if env.is_terminal(state):
                continue

            v = state_values[state[0], state[1]]
            chosen_action = self.policy(state)
            dynamics = env.dynamics(state, chosen_action)

            q_pi = 0.0
            for new_state, rewards in dynamics.items():
                for reward, p in rewards.items():
                    v_s_prime = state_values[new_state[0], new_state[1]]
                    q_pi += p * (reward + self.discount_factor*v_s_prime) # q_π(s,a) = Σ p(s',r|s,a)[r + 𝛾V(s')]
            v_s = 1.0 * q_pi # deterministic
            delta = max(delta, abs(v - v_s))
            state_values[state[0], state[1]] = v_s
        return state_values, delta
    
    def policy_evaluation(self, env : Environment, iterations : int=1000, epsilon : float=1e-5):
        for _ in tqdm(range(iterations)):
            new_state_values, delta = self.sweep(env, self.state_values.copy())
            if delta < epsilon:
                break
            self.state_values = new_state_values

if __name__ == "__main__":
    N = 4
    DISCOUNT_FACTOR = 0.5
    env = Environment(N)
    agent = Agent(env, DISCOUNT_FACTOR)

    print("Terminal states:")
    print(env.terminal_states)

    print("Initial state values:")
    print(agent.state_values)

    print("Policy evaluation...")
    agent.policy_evaluation(env)

    print("Result state values:")
    print(agent.state_values)