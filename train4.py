import math
import time
import random
import json

from collections import namedtuple, deque
from itertools import count

from env import *

import pyautogui

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

level_data = get_file_binary("Stageix")

loss_arr = []
reward_arr = []

BEST_X = 0

BATCH_SIZE = len(view()) + len(level_data)
print("Batch size:", BATCH_SIZE)
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005  # 0.005
LR = 1e-2 #adapt learning rate to BEST_X 

states = {"INIT": 0, "PLAY": 1, "WIN": 2, "LOSE": 3}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    # 10 actions per second, 600 per minutes
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(BATCH_SIZE, 1000)
        self.layer2 = nn.Linear(1000, 60)
        self.layer3 = nn.Linear(60, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


###############

# # Create an instance of the DQN model
# policy_net = DQN()

# # Generate a random tensor of size 50,000
# input_data = torch.randn(BATCH_SIZE)

# # Evaluate the policy_net DQN with the random tensor 10 times and measure the time
# total_time = 0
# for _ in range(10):
#     start_time = time.time()
#     output = policy_net(input_data)
#     end_time = time.time()
#     total_time += (end_time - start_time)

# # Calculate the average time taken per evaluation
# average_time = total_time / 10

# print(f"Average time taken per evaluation: {average_time} seconds")

###############

device = torch.device("cuda")

print("Device:", device)

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

print(target_net.layer1.weight)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(BATCH_SIZE)

def mutate_model(model, lr):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * lr)

def optimize_model():
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
    
    loss_arr.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def act(arr):
    time.sleep(1)

    print("Acting")
    
    pyautogui.click(x=800, y=1300)

    # print(arr)
    for i in range(len(arr)):
        if arr[i] == 1:
            pyautogui.click(x=pyautogui.size()[0]/2, y=pyautogui.size()[1]/2)
        if arr[i:] == [0] * len(arr[i:]) or state_from_screen() != "PLAY":
            return
        time.sleep(0.10)
    return


# level_data = level_data + [0]*(BATCH_SIZE - len(level_data))
# input_data = torch.tensor(level_data, device=device)

# print("Input data shape:", input_data.shape)

time.sleep(3)

epoch = 0

while state_from_screen() != "WIN":
    # print(policy_net.layer1.weight[0])
    
    print(f"Epoch {epoch}")
    
    epoch += 1

    # Respawn
    if state_from_screen() == "LOSE":
        pyautogui.click(x=800, y=1300)

    # Wait for the state to change
    print("Ready to act")
    while (state := state_from_screen()) == "PLAY":
        
        input_data = torch.tensor(level_data + view(), device=device)
        
        # Perform forward pass through policy_net
        output = policy_net(input_data)

        # Convert output to list
        output_list = output.squeeze().tolist()
        # print(output_list)

        if output_list > 0:
            pyautogui.click(x=pyautogui.size()[0]/2, y=pyautogui.size()[1]/2)

    # Calculate reward based on the state transition
    if state == "WIN":
        reward = 1
        BEST_X = 0
    else:
        t, att, x, y = transition()
        if x > BEST_X:
            reward = (x - BEST_X) / x
            BEST_X = x
        else:
            reward = (x - BEST_X) / BEST_X

    print(f"Reward: {reward}")

    # Convert reward to tensor
    reward = torch.tensor([reward], device=device)

    # Convert current state to tensor
    state_tensor = torch.tensor(states[state], dtype=torch.float32, device=device).unsqueeze(0)

    # Check if it's a terminal state
    done = state == "WIN"

    # Set next state
    if done:
        next_state = None
    else:
        next_state = torch.tensor(states["PLAY"], dtype=torch.float32, device=device).unsqueeze(0)

    # Push current transition to memory
    memory.push(state_tensor, torch.tensor(output_list, device=device), next_state, reward)
    
    reward_arr.append(reward.item())

    # Optimize model
    optimize_model()
    
    # Write new object to history.json
    history = {
        "epoch": epoch,
        "reward": reward_arr[-1] if len(reward_arr) > 0 else 0,
        "loss": loss_arr[-1] if len(loss_arr) > 0 else 0
    }

    with open("history.json", "a") as file:
        file.write(json.dumps(history) + "\n")
    
    # Mutate the model
    mutate_model(policy_net, LR)

    # Update target network
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(policy_param.data * TAU + target_param.data * (1.0 - TAU))

    if done:
        break
