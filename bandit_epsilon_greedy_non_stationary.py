import numpy as np
from tqdm import tqdm

k = 10  # Number of arms
epsilon = 0.1  # Exploration rate
iteration = 100000  # Number of iterations
mean_shift = 0.01  # Amount to shift means at each update
std_shift = 0.005  # Amount to shift standard deviations at each update

Q = [0 for _ in range(k)]  # Estimated action values
N = [0 for _ in range(k)]  # Count of selections for each action

class Bandit:
    def __init__(self, k, min_mean=-1, max_mean=1, min_std=1, max_std=2):
        # Initialize mean and standard deviation for each arm
        self.mean = np.random.uniform(min_mean, max_mean, size=k)
        self.std = np.random.uniform(min_std, max_std, size=k)
    
    def __call__(self, a):
        # Generate a reward from a normal distribution for the chosen action
        return np.random.normal(self.mean[a], self.std[a])
    
    def update_distribution(self):
        # Randomly shift each mean and standard deviation slightly
        self.mean += np.random.normal(0, mean_shift, size=k)
        
        # Ensure std stays positive after update
        self.std = np.clip(self.std + np.random.normal(0, std_shift, size=k), 0.1, None)
    
    def print(self):
        print("Mean:")
        print(self.mean)
        print("Std:")
        print(self.std)

bandit = Bandit(k)

for i in tqdm(range(iteration)):
    # Choose action (explore with probability epsilon, otherwise exploit)
    if np.random.rand() < epsilon:
        index = np.random.randint(0, k)
    else:
        index = np.argmax(Q)
    
    # Get reward for the selected action
    R = bandit(index)
    
    # Update action counts and estimated values using incremental update formula
    N[index] += 1
    Q[index] += (1 / N[index]) * (R - Q[index])
    
    # Periodically update the means and stds to simulate a non-stationary environment
    if i % 100 == 0:  # Update distribution parameters every 100 iterations
        bandit.update_distribution()

# Print final results
print("Q values:")
print(Q)
print("N values:")
print(N)
bandit.print()
