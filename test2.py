import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.policy_net = DQN(input_size, hidden_size, output_size)
        self.target_net = DQN(input_size, hidden_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.output_size)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()

    def train(self, state, action, next_state, reward, done):
        self.optimizer.zero_grad()
        q_values = self.policy_net(state)
        target_q_values = q_values.clone().detach()

        if done:
            target_q_values[action] = reward
        else:
            with torch.no_grad():
                target_q_values[action] = reward + self.gamma * torch.max(self.target_net(next_state))

        loss = self.loss_fn(q_values, target_q_values.unsqueeze(0))
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

input_size = 3000000
hidden_size = 30
output_size = 30

agent = Agent(input_size, hidden_size, output_size)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32)
        agent.train(state, action, next_state, reward, done)
        total_reward += reward.item()
        state = next_state

    if episode % target_update_frequency == 0:
        agent.update_target_net()
