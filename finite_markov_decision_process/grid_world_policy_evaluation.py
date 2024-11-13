import numpy as np
from tqdm import tqdm
from typing import Tuple, List
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
        self.terminal_states = [(0, 0), (self.world_size[0]-1, self.world_size[1]-1)]
    
    def is_terminal(self, state : Tuple[int]):
        return state in self.terminal_states
    
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
        return new_state, reward

class Agent:
    def __init__(self, env : Environment, discount_factor : int=0.5):
        self.state_values = self.init_state_values(env) # V(s)
        self.current_state = (
            np.random.randint(low=0, high=env.world_size[0]),
            np.random.randint(low=0, high=env.world_size[1])
        )
        self.discount_factor = discount_factor
    
    def init_state_values(self, env : Environment):
        state_values = np.random.rand(*env.world_size)
        for terminal_state in env.terminal_states:
            state_values[*terminal_state] = 0.0
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

    def get_policy(self, state : Tuple[int], actions : List[int]): # π(a|s), but for now it is deterministic. So, we ignore the state
        return [1/(len(actions)) for _ in actions]
    
    def action(self, state : Tuple[int], env : Environment):
        actions = self.available_actions(state, env)
        pi = self.get_policy(state, actions)
        action = np.random.choice(actions, p=pi)
        return action
    
    def sweep(self, env : Environment, state_values : np.ndarray):
        for i in range(state_values.shape[0]):
            for j in range(state_values.shape[1]):
                state = (i, j)
                if env.is_terminal(state):
                    continue

                actions = self.available_actions(state, env)
                pi = self.get_policy(state, actions)

                v_s = 0.0
                for a, prob in zip(actions, pi):
                    new_state, reward = env.dynamics(state, action=a)
                    v_s_prime = state_values[*new_state]
                    v_s += prob * (reward + self.discount_factor*v_s_prime) # V(s) = Σ π(a|s) Σ p(s',r|s,a)[r + 𝛾V(s')] , since the reward and the next state is deterministic, the calculation is simple
                
                state_values[*state] = v_s
        return state_values
    
    def policy_evaluation(self, env : Environment, iterations : int=1000, epsilon : float=1e-5):
        for _ in tqdm(range(iterations)):
            new_state_values = self.sweep(env, self.state_values.copy())
            l2 = ((new_state_values - self.state_values)**2).mean()
            if l2 <= epsilon:
                break
            self.state_values = new_state_values

if __name__ == "__main__":
    N = 6
    DISCOUNT_FACTOR = 0.5
    env = Environment(N)
    agent = Agent(env, DISCOUNT_FACTOR)

    print("Initial state values:")
    print(agent.state_values)

    print("Policy evaluation...")
    agent.policy_evaluation(env)

    print("Result state values:")
    print(agent.state_values)