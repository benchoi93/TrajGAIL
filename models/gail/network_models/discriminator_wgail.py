# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# import gym
from torch.nn.utils.rnn import pack_padded_sequence
from models.gail.network_models.discriminator_pytorch import Discriminator as Discrim_vanilla
from models.gail.network_models.policy_net_rnn import StateSeqEmb

class Discriminator(Discrim_vanilla):
    def __init__(self, state_dim, action_dim, hidden:int, disttype = "categorical", num_layers = 3):
        super(Discriminator, self).__init__(state_dim, action_dim, hidden, disttype )
        self.num_layers = num_layers
        self.StateSeqEmb = StateSeqEmb(self.state_dim,self.action_dim, self.hidden,num_layers)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.2 , inplace=True)

    def forward(self, state_seq, action, seq_len):
        # state_seq = exp_obs
        # action = exp_act
        # seq_len = exp_len
        # state_seq[state_seq == -1] = self.state_dim
        _, x_rnn = self.StateSeqEmb(state_seq, seq_len)         
        x_rnn = x_rnn[self.num_layers-1,:,:]
        # state = self.one_hotify(state, self.state_dim)
        action = self.one_hotify(action.unsqueeze(1), self.action_dim)
        
        x = torch.cat([x_rnn, action], dim = 1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        prob =  self.fc4(x) 
        return prob


    def get_reward(self,*args,**kwargs):
        with torch.no_grad():
                if kwargs:
                    prob = self.forward(**kwargs)
                elif args:
                    prob = self.forward(*args)
                else:
                    raise ValueError
                return prob
    

