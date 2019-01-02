import gym
import numpy as np
from tqdm import * 
from matplotlib import pyplot as plt

gamma = 1 # discounting

#actor
actor_alpha = 0.05 # learning rate
actor_w = np.random.rand(2, 4)

#critic
critic_alpha = 0.05 # learning rate
critic_w = np.random.rand(2, 4)

def Q(st, ac=None):
   if ac is None:
      return np.dot(critic_w, st).ravel()
   return np.dot(critic_w[ac], st)[0]

#softmax policy
def softmax(st):
   z = np.dot(actor_w, st).ravel()
   exp = np.exp(z)
   return exp / np.sum(exp)

def policy(st):
   return np.random.choice([0, 1], p=softmax(st))

def eligibility_vector(st):
   p = softmax(st)
   s = np.sum([p[i] * st for i in range(2)])
   return st - s

def normalize(st):
   player_hand = (st[0] - 4) / 25
   dealer_card = (st[1] - 2) / 9
   return np.array([player_hand, dealer_card, player_hand * int(st[2]), 1]).reshape(4, 1)

env = gym.make('Blackjack-v0')

episodes = 10
win = 0
lose = 0
draw = 0
ave_reward = []

# reward: win = 1, draw = 0, lose = 0, not reward for intermediate steps
for e in tqdm(range(episodes)):
   done = False
   state = normalize(env.reset())
   action = policy(state)
   total_reward = 0
   while not done:
      new_state, reward, done, _ = env.step(action)
      total_reward += reward

      if done:
         td_target = reward
         if 1 == reward:
            win += 1
         elif -1 == reward:
            lose += 1
         else:
            draw += 1
         ave_reward.append(reward / (e + 1))

      else:
         new_state = normalize(new_state)
         new_action = policy(new_state)
         #reward is 0, can be omitted
         td_target = gamma * Q(new_state, new_action)

      td_error = td_target - Q(state, action)

      print('ev = ', eligibility_vector(state))

      #update critic
      critic_w[action] += critic_alpha * td_error * state

      #update actor
      #actor_w[action] += actor_alpha * gamma * td_error *
