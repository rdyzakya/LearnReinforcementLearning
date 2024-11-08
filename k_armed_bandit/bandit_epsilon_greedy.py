import numpy as np
from tqdm import tqdm

k = 10
epsilon = 0.1
iteration = 100000

Q = [0 for _ in range(k)]
N = [0 for _ in range(k)]

class Bandit:
    def __init__(self, k, min_mean=-1, max_mean=1, min_std=1, max_std=2):
        self.mean = [np.random.uniform(min_mean, max_mean) for _ in range(k)]
        self.std = [np.random.uniform(min_std, max_std) for _ in range(k)]
    
    def __call__(self, a):
        return np.random.normal(self.mean[a], self.std[a])
    
    def print(self):
        print("Mean:")
        print(self.mean)
        print("Std:")
        print(self.std)

bandit = Bandit(k)

for i in tqdm(range(iteration)):
    if np.random.rand() < epsilon:
        index = np.random.randint(0, k)
    else:
        index = np.argmax(Q)

    R = bandit(index)
    N[index] += 1
    Q[index] += (1/N[index]) * (R - Q[index])

print("Q:")
print(Q)
print("N:")
print(N)
bandit.print()