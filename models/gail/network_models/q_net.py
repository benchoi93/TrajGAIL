# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from network_models.policy_net_rnn import StateSeqEmb


class Posterior_net(nn.Module):
    def __init__(self, state_dim, encode_dim ,hidden:int,origins,start_code):
        super(Posterior_net, self).__init__()
        self.state_dim = state_dim
        self.encode_dim = encode_dim
        self.hidden = hidden

        self.state_emb = nn.Embedding(self.state_dim , self.hidden)

        self.fc1 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc2 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc3 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc4 = nn.Linear(in_features = self.hidden , out_features=self.encode_dim)

        self.activation = torch.relu

    def forward(self, state):
        x = self.state_emb(state)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.log_softmax(self.fc4(x), dim = 1)
        
        action_dist = torch.distributions.Categorical(x.exp())
        return action_dist
