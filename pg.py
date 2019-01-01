import gym
import numpy as np
from tqdm import * 
from matplotlib import pyplot as plt

#actor
actor_alpha = 0.05 # learning rate
actor_w = np.random.rand(2, 4)

#critic
critic_alpha = 0.05 # learning rate
critic_gamma = 1 # discounting
critic_w = np.random.rand(2, 4)

def normalize(st):
   player_hand = (st[0] - 4) / 25
   dealer_card = (st[1] - 2) / 9
   return np.array([player_hand, dealer_card, player_hand * int(st[2]), 1]).reshape(4, 1)

env = gym.make('Blackjack-v0')

episodes = 1

for e in tqdm(range(episodes)):
   done = False
   state = normalize(env.reset())
