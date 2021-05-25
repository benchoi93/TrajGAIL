# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# import gym
from torch.nn.utils.rnn import pack_padded_sequence
from models.gail.network_models.discriminator_pytorch import Discriminator as Discrim_vanilla
from models.gail.network_models.policy_net_rnn import StateSeqEmb



class infoQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden:int, disttype = "categorical"):
        super(infoQ, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.disttype = disttype

        self.hidden = hidden
        self.disttype = disttype
        self.state_emb = nn.Embedding(self.state_dim , self.hidden)

        self.activation = torch.relu

        self.fc1 = nn.Linear(in_features = self.hidden+self.action_dim, out_features=self.hidden)
        self.fc2 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc3 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc4 = nn.Linear(in_features = self.hidden , out_features=1)

    def one_hotify(self, longtensor, dim):
        if list(self.parameters())[0].is_cuda:
            one_hot = torch.cuda.FloatTensor(longtensor.size(0) , dim).to()
        else:
            one_hot = torch.FloatTensor(longtensor.size(0) , dim).to()
        one_hot.zero_()
        one_hot.scatter_(1,longtensor.long(),1)
        return one_hot

    def forward(self, state, action):
        # state = self.one_hotify(state, self.state_dim)
        state = self.state_emb(state)
        action = self.one_hotify(action.unsqueeze(1), self.action_dim)
        # state =  self.state_emb(state)
        x = torch.cat([state, action], dim = 1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        prob = torch.sigmoid(self.fc4(x) )
        return prob

def sample_gumbel(n,k):
    unif = torch.distributions.Uniform(0,1).sample((n,k))
    g = -torch.log(-torch.log(unif))
    return g

def sample_gumbel_softmax(pi, n, temperature):
    k = len(pi)
    g = sample_gumbel(n, k)
    h = (g + torch.log(pi))/temperature
    h_max = h.max(dim=1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y


class infoQ_RNN(infoQ):
    def __init__(self, state_dim:int, action_dim:int,hidden:int, num_outs:int, disttype = "categorical", num_layers = 3, probs=None):
        super(infoQ_RNN, self).__init__(state_dim, action_dim, hidden, disttype )
        self.num_layers = num_layers
        self.num_outs = num_outs
        self.tau = 0.1

        self.StateSeqEmb = StateSeqEmb(self.state_dim,self.action_dim, self.hidden,num_layers)
        self.fc4 = nn.Linear(in_features = self.hidden , out_features=self.num_outs)
        
        # a = torch.distributions.relaxed_categorical.ExpRelaxedCategorical(self.tau, probs = self.prob)
        ##TODO :: logit

        # (r1 - r2) * torch.rand(self.num_outs) + r2

        self.logit=None
        self.probs=None

        if probs is not None:
            self.probs=torch.nn.Parameter(torch.Tensor(probs),requires_grad=False)
        else:
            logit = torch.zeros((self.num_outs) , dtype = torch.float)
            self.logit = torch.nn.Parameter(logit)
            torch.nn.init.uniform_(self.logit)            

    def encode(self,n):
        if self.logit is not None:
            self.encode_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(self.tau, logits = self.logit)
        elif self.probs is not None:
            self.encode_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(self.tau, probs = self.probs)
        else:
            raise ValueError

        # self = p0
        # n=100
        return self.encode_dist.rsample(torch.Size((n,)) )

        # result = []
        # for _ in range(n):
        #     result.append(self.encode_dist.rsample().unsqueeze(0))
        # return torch.cat(result , axis =0)

    def forward(self, state_seq, action_prob, seq_len):
        # state_seq = learner_obs
        # action = learner_act
        # seq_len = learner_len

        # state_seq[state_seq == -1] = self.state_dim
        _, x_rnn = self.StateSeqEmb(state_seq, seq_len)         
        x_rnn = x_rnn[self.num_layers-1,:,:]
        # state = self.one_hotify(state, self.state_dim)
        # action = action_prob
        
        x = torch.cat([x_rnn, action_prob], dim = 1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
