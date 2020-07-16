# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# import gym





class Policy_net(nn.Module):
    def __init__(self, state_dim, action_dim ,hidden:int,origins,start_code, disttype = "categorical"):
        super(Policy_net, self).__init__()
        """
        :param name: string
        :param env: gym env
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.disttype = disttype

        self.origins = origins
        self.origin_dim = origins.shape[0]
        self.start_code = start_code

        self.state_emb = nn.Embedding(self.state_dim , self.hidden)

        self.fc1 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc2 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc3 = nn.Linear(in_features = self.hidden , out_features=self.hidden)

        self.activation = torch.relu

        if disttype == "categorical":
            self.fc4 = nn.Linear(in_features = self.hidden , out_features=self.action_dim)
            self.origin_fc4 = nn.Linear(in_features = self.hidden , out_features=self.origin_dim)
        elif disttype == "normal":
            self.policy_mu = nn.Linear(in_features = self.hidden , out_features=self.action_dim)
            self.policy_sigma = nn.Linear(in_features = self.hidden , out_features=self.action_dim)
        else:
            raise ValueError

        self.prob_dim = max(self.action_dim, self.origin_dim)
    
    def forward(self, state):
        x = self.state_emb(state)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        origins_idx = state == self.start_code

        origins_prob = torch.softmax(self.origin_fc4(x[origins_idx]) , dim = 1)
        others_prob  = torch.softmax(self.fc4(x[~origins_idx]) , dim = 1)
        
        # prob = torch.softmax(self.fc4(x) , dim = 1)
        prob = torch.zeros(size = (state.shape[0] , self.prob_dim) , device = x.device)
        prob[origins_idx] = origins_prob
        prob[~origins_idx,:self.action_dim] = others_prob

        action_dist = torch.distributions.Categorical(prob)
        return action_dist

    def act(self, *args , **kwargs):
    # def act(self, state, stochastic= True):
        if kwargs:
            dist = self.forward(**kwargs)
        elif args:
            dist = self.forward(*args)
        action = dist.sample().item()
        return action

class Value_net(nn.Module):
    def __init__(self,state_dim,action_dim , hidden:int):
        super(Value_net, self).__init__()
        """
        :param name: string
        :param env: gym env
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.state_emb = nn.Embedding(self.state_dim , self.hidden)
        self.activation = torch.relu

        self.fc1 = nn.Linear(in_features = self.hidden + self.action_dim, out_features=self.hidden)
        self.fc2 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc3 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc4 = nn.Linear(in_features = self.hidden , out_features=1)
    
    def one_hotify(self, longtensor, dim):
        if list(self.parameters())[0].is_cuda:
            one_hot = torch.cuda.FloatTensor(longtensor.size(0) , dim).to()
        else:
            one_hot = torch.FloatTensor(longtensor.size(0) , dim).to()
        one_hot.zero_()
        one_hot.scatter_(1,longtensor.unsqueeze(1).long(),1)
        return one_hot
    # def forward(self, state):
    def forward(self, state, action):
        state = self.state_emb(state)
        action = self.one_hotify(action, self.action_dim)
        x = torch.cat([state, action], dim = 1)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        value = self.fc4(x)
        return value





