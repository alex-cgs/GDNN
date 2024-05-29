import torch
import torch.nn as nn
import torch.nn.functional as F

n_observations = 2000000

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 30)
        self.layer2 = nn.Linear(30, 30)
        self.layer3 = nn.Linear(30, n_actions)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.layer3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = DQN(2000000, 30).to(device)



# policy




def train(dqn):
    
    # FOR EACH TRIAL
    
    # GATHER ALL INP DATA
    
    # FWD IN DQN
    
    # LOSS
    
    
    return