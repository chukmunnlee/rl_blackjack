import gym
import numpy as np
from tqdm import *
from matplotlib import pyplot as plt

#player hand, dealer, usable ace
#(15, 5, True)

w = np.random.rand(2, 4)
#w = np.ones(shape=(2, 4))

epsilon = 0.01 # explore/exploit
alpha = 0.05 # learning rate
gamma = 1 # discount
episodes = 500000

decay = 1 / episodes

def Q(st, ac=None):
   if ac is None:
      return np.dot(w, st).ravel()
   return np.dot(w[ac], st)[0]

def policy(st):
   if np.random.rand() > epsilon:
      return np.argmax(Q(st))
   return np.random.randint(0, 2)

def normalize(st):
   player_hand = (st[0] - 4) / 25
   dealer_card = (st[1] - 2) / 9
   return np.array([player_hand, dealer_card, player_hand * int(st[2]), 1]).reshape(4, 1)

env = gym.make('Blackjack-v0')

win = 0
lose = 0
draw = 0
total_reward = 0
winlose = []
ave_reward = []

for e in tqdm(range(episodes)):

   state = normalize(env.reset())
   action = policy(state)
   done = False

   while not done:
      new_state, reward, done, _ = env.step(action)
      if done:
         total_reward += reward
         ave_reward.append(total_reward / (e + 1))
         td_target = reward
         new_action = 0
         winlose.append(reward + 2)
         if -1 == reward:
            lose += 1
         elif 0 == reward:
            draw += 1
         else:
            win += 1
      else:
         new_state = normalize(new_state)
         new_action = policy(new_state)
         td_target = reward + (gamma * Q(new_state, new_action))

      td_error = td_target - Q(state, action)
      w[action] += (alpha * td_error * state).T.ravel()

      state = new_state
      action = new_action

   epsilon = max(epsilon - decay, 0.005)
   #alpha = max(alpha - decay, 0.005)

print('win: %.2f, lose: %.2f, draw: %.2f' %(win/episodes, lose/episodes, draw/episodes))

winlose = np.array(winlose)

fig = plt.figure()

ax = fig.add_subplot(121)
ax.set_title('Win, Lose, Draw')

labels = ['Lose', 'Draw', 'Win']
for i in range(1, len(labels) + 1):
   idx = np.where(winlose == i)
   ax.scatter(idx, winlose[idx], label=labels[i - 1], s=0.5)
plt.legend()

ax = fig.add_subplot(122)
ax.set_title('Average reward')
ax.plot(range(episodes), ave_reward, label='Ave reward')
plt.legend()

plt.show()
