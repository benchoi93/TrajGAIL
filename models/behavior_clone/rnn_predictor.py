
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN_predictor(nn.Module):
    def __init__(self, states , hidden, pad_idx = -1,num_layers=2):
        super(RNN_predictor,self).__init__()
        self.hidden = hidden
        self.pad = pad_idx
        self.states = states +[self.pad]
        self.num_layers=num_layers

        self.state_dim= len(self.states)
        self.state_emb = nn.Embedding(self.state_dim , self.hidden)
        self.rnncell = nn.LSTM(self.hidden, self.hidden, self.num_layers, batch_first = True)
        self.linear1 = nn.Linear(self.hidden,self.hidden)
        self.linear2 = nn.Linear(self.hidden,self.state_dim)

        self.find_idx = lambda x : self.states.index(x)
        self.np_find_idx = np.vectorize(self.find_idx)
        self.pad_idx = self.find_idx(self.pad)

        self.device = torch.device("cpu")

    def states_to_idx(self, states_seq:np.array):
        idx_seq = self.np_find_idx(states_seq)
        return idx_seq

    def init_hidden(self,batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers,batch_size,self.hidden)).to(self.device)
        cell = Variable(torch.zeros(self.num_layers,batch_size,self.hidden)).to(self.device)
        return hidden, cell

    def forward(self, x_train):
        x = self.state_emb(x_train)

        # hidden,cell = self.init_hidden(x_train.size(0))
        output, (hidden,cell) = self.rnncell(x)

        y_est = F.relu(self.linear1(output) )
        # y_est = F.softmax(self.linear2(y_est) , dim = 2)
        y_est = F.log_softmax(self.linear2(y_est), dim =2)
        return y_est

    def unroll_trajectories(self,start_state, end_state,num_trajs, max_length = 20):
        # start_state = sw.start
        # end_state = sw.terminal
        # num_trajs = 200
        # max_length = 20

        end_idx = self.find_idx(end_state)
        learner_trajs  = torch.ones((num_trajs, 1)).long().to(self.device) * self.find_idx(start_state)
        done_mask = torch.zeros((num_trajs)).bool().to(self.device)

        for i in range(max_length):
            if done_mask.sum() == num_trajs:
                break

            input_trajs = learner_trajs[~done_mask,:]

            # next_prob = F.softmax(self.forward(input_trajs.to(self.device) ) , dim = 2)
            next_prob = self.forward(input_trajs.to(self.device) ).exp()
            next_prob_dist = torch.distributions.Categorical(next_prob[:,-1,:(self.state_dim-1)])
            next_idx = next_prob_dist.sample()
            next_idx.unsqueeze_(1)
            
            next_idx_whole = torch.ones((num_trajs, 1)).long().to(self.device) * self.pad_idx
            next_idx_whole[~done_mask] = next_idx

            learner_trajs = torch.cat([learner_trajs, next_idx_whole],dim=1)       
            is_done = (next_idx_whole == end_idx) | (next_idx_whole == self.pad_idx)

            done_mask = done_mask | is_done.view(done_mask.size())
        return learner_trajs