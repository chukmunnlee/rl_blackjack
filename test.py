import numpy as np

w = np.random.rand(2, 4)
state = np.random.rand(1, 4).ravel()

z = state.dot(w.T)
exp = np.exp(z)
prob = exp / np.sum(exp)

print('z = ', z)
print('exp = ', exp)

print('softmax = ', prob)

for i in range(10):
   print(np.random.choice(np.arange(2), p=prob))
