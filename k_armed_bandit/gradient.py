import numpy as np
from tqdm import tqdm

k = 10
alpha = 0.1
iteration = 100000
Rt = 0
Rt_ = 0

Ht = np.zeros(k)
N = [0 for _ in range(k)]

def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator)
    return numerator / denominator

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
    pi = softmax(Ht)
    # index = np.argmax(pi)
    index = np.random.choice(range(k), p=pi)

    Rt = bandit(index)
    Rt_ = (Rt + Rt_*i)/(i+1)
    for j in range(k):
        if j == index:
            Ht[j] += alpha * (Rt - Rt_)*(1 - pi[j])
        else:
            Ht[j] -= alpha * (Rt - Rt_)*pi[j]
    N[index] += 1

print("Ht:")
print(Ht)
print("N:")
print(N)
bandit.print()