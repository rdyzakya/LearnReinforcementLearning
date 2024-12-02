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
        return max(action_values, key=action_values.get)
    
    def sweep(self, env : Environment, state_values : np.ndarray):
        states = env.get_all_states()
        delta = 0.0
        for state in states:
            if env.is_terminal(state):
                continue

            v = state_values[state[0], state[1]]
            action = self.policy(state)
            v_s = 0.0

            dynamics = env.dynamics(state, action)
            q_pi = 0.0
            for new_state, rewards in dynamics.items():
                for reward, p in rewards.items():
                    v_s_prime = state_values[new_state[0], new_state[1]]
                    q_pi += p * (reward + self.discount_factor*v_s_prime) # q_π(s,a) = Σ p(s',r|s,a)[r + 𝛾V(s')]
            v_s += 1.0 * q_pi # V(s) = Σ π(a|s)q_π(s,a) but deterministic

            delta = max(delta, abs(v - v_s))
            state_values[state[0], state[1]] = v_s
        return state_values, delta
    
    def policy_evaluation(self, 
                          env : Environment, 
                          state_values : np.ndarray, 
                          iterations : int=1000, 
                          epsilon : float=1e-5):
        for _ in tqdm(range(iterations), desc="Policy Evaluation"):
            new_state_values, delta = self.sweep(env, state_values.copy())
            if delta < epsilon:
                break
            state_values = new_state_values
        return state_values
    
    def policy_improvement(self, env : Environment, state_action_values : Dict):
        states = env.get_all_states()
        policy_stable = True
        for state in states:
            if env.is_terminal(state):
                continue

            available_actions = []
            old_action_values = []

            for a, v in state_action_values[state].items():
                available_actions.append(a)
                old_action_values.append(v)
            
            max_old_action_values = max(old_action_values)

            old_actions = set()

            for i in range(len(available_actions)):
                if old_action_values[i] == max_old_action_values:
                    old_actions.add(available_actions[i])
            
            new_action_values = []

            for a in available_actions:
                dynamics = env.dynamics(state, a)
                q_pi = 0.0
                for new_state, rewards in dynamics.items():
                    for reward, p in rewards.items():
                        v_s_prime = self.state_values[new_state[0], new_state[1]]
                        q_pi += p * (reward + self.discount_factor*v_s_prime) # q_π(s,a) = Σ p(s',r|s,a)[r + 𝛾V(s')]
                new_action_values.append(q_pi)
            
            max_new_action_values = max(new_action_values)

            new_actions = set()

            for i in range(len(available_actions)):
                if new_action_values[i] == max_new_action_values:
                    new_actions.add(available_actions[i])
                
                state_action_values[state][available_actions[i]] = new_action_values[i] # update state action values
            
            if len(old_actions.difference(new_actions)) != 0: # we can use something like delta in policy evaluation if we want
                policy_stable = False

        return state_action_values, policy_stable
    
    
    def policy_iteration(self, 
                        env : Environment, 
                        iterations : int=1000, 
                        evaluation_iterations : int=1000,
                        evaluation_epsilon : float=1e-5):
        for _ in tqdm(range(iterations), desc="GPI"):
            new_state_values = self.policy_evaluation(env, 
                                                      self.state_values.copy(),
                                                      evaluation_iterations,
                                                      evaluation_epsilon)
            self.state_values = new_state_values

            self.state_action_values, policy_stable = self.policy_improvement(env,
                                                                             deepcopy(self.state_action_values))
            if policy_stable:
                break


if __name__ == "__main__":
    N = 5
    DISCOUNT_FACTOR = 0.5
    env = Environment(N)
    agent = Agent(env, DISCOUNT_FACTOR)

    print("Terminal states:")
    print(env.terminal_states)

    print("Initial state values:")
    print(agent.state_values)

    print("GPI...")
    agent.policy_iteration(env)

    print("Result state values:")
    print(agent.state_values)