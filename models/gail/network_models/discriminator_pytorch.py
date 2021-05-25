# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# import gym


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden: int, disttype="categorical"):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.disttype = disttype

        self.hidden = hidden
        self.disttype = disttype
        self.state_emb = nn.Embedding(self.state_dim, self.hidden)

        self.activation = torch.relu

        self.fc1 = nn.Linear(in_features=self.hidden +
                             self.action_dim, out_features=self.hidden)
        self.fc2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc3 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc4 = nn.Linear(in_features=self.hidden, out_features=1)

    def one_hotify(self, longtensor, dim):
        if list(self.parameters())[0].is_cuda:
            one_hot = torch.cuda.FloatTensor(longtensor.size(0), dim).to()
        else:
            one_hot = torch.FloatTensor(longtensor.size(0), dim).to()
        one_hot.zero_()
        one_hot.scatter_(1, longtensor.long(), 1)
        return one_hot

    def forward(self, state, action):
        # state = self.one_hotify(state, self.state_dim)
        state = self.state_emb(state)
        action = self.one_hotify(action.unsqueeze(1), self.action_dim)
        # state =  self.state_emb(state)
        x = torch.cat([state, action], dim=1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        prob = torch.sigmoid(self.fc4(x))
        return prob

    def get_reward(self, *args, **kwargs):
        with torch.no_grad():
            if kwargs:
                prob = self.forward(**kwargs)
            elif args:
                prob = self.forward(*args)
            else:
                raise ValueError
            return -torch.log(torch.clamp(prob, 1e-10, 1))
            # p = torch.clamp(prob , 1e-10 , 1)
            # return -(p.log() + (1-p).log())
