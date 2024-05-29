import math
import random
import asyncio

from collections import namedtuple, deque
from itertools import count

from env import *

import pyautogui

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import numpy as np
# w1 = np.random.uniform(-1, 1, size=(2000000, 20))
# w2 = w1 = np.random.uniform(-1, 1, size=(20, 20))
# print(w1.dot(w2))

global actions
global current_state
global training
actions = []
current_state = "init"
training = True

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(2000000, 60)
        self.layer2 = nn.Linear(60, 60)
        self.layer3 = nn.Linear(60, 30)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

async def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

async def act():
    while training:
        if current_state == "play":
            if actions != []:
                act = actions[0]
                actions = actions[1:]
                
                if act == 1:
                    pyautogui.press('space')
        await asyncio.sleep(1/60)

record = []

level_data = get_file_binary(1)
level_data = level_data + [0]*(2000000 - len(level_data) - 4)

action = asyncio.create_task(act())

#training loop

async def main():
    for epoch in range(500):
        print(epoch)
        
        actions = []  # Reset actions for each epoch
        current_state = "init"  # Reset current state for each epoch
        
        new_data = transition()
        input_data = torch.tensor(new_data + level_data)
        
        if record and record[-1][1] == new_data[1] and record[-1][2] == new_data[2] and record[-1][3] == new_data[3]:
            current_state = "dead"
        else:
            current_state = "play"
        
        record.append(new_data)
        
        todo = list(policy_net.forward(input_data))
        
        for i in range(len(todo)):
            todo[i] = round(todo[i])
        
        actions += todo

        await optimize_model()  # Await the optimization step

main()