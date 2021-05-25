# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# import gym


class Policy_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden: int, origins, start_code, env, disttype="categorical"):
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
        self.env = env

        self.state_emb = nn.Embedding(self.state_dim, self.hidden)

        self.fc1 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc3 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.activation = torch.relu

        # self.fc4 = nn.Linear(in_features=self.hidden,
        #  out_features=self.action_dim)
        self.origin_fc4 = nn.Linear(
            in_features=self.hidden, out_features=self.origin_dim)

        self.prob_dim = max(self.action_dim, self.origin_dim)

        self.action_domain = torch.zeros(
            (len(self.env.states), self.env.max_actions)).long()
        for i in range(len(self.env.states)):
            s0 = self.env.states[i]
            if not s0 == self.env.terminal:
                self.action_domain[i, list(self.env.netconfig[s0].keys())] = 1
        self.action_domain = torch.nn.Parameter(
            self.action_domain, requires_grad=False)

    def forward(self, state):
        x = self.state_emb(state)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        masked_logit = self.origin_fc4(x).masked_fill(
            (1-self.action_domain[state]).bool(), -1e32)
        prob = torch.nn.functional.softmax(masked_logit, dim=1)
        action_dist = torch.distributions.Categorical(prob)

        return action_dist


class Value_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden: int):
        super(Value_net, self).__init__()
        """
        :param name: string
        :param env: gym env
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
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
        one_hot.scatter_(1, longtensor.unsqueeze(1).long(), 1)
        return one_hot
    # def forward(self, state):

    def forward(self, state, action):
        state = self.state_emb(state)
        action = self.one_hotify(action, self.action_dim)
        x = torch.cat([state, action], dim=1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        value = self.fc4(x)
        return value
