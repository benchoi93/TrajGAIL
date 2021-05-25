# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# import gym
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.gail.network_models.policy_net_pytorch import Policy_net as Policy_vanilla
from models.gail.network_models.policy_net_pytorch import Value_net as Value_vanilla


class StateSeqEmb(nn.Module):
    def __init__(self, state_dim, action_dim, hidden, num_layers):
        super(StateSeqEmb, self).__init__()
        self.state_dim = state_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.action_dim = action_dim

        self.state_emb = nn.Embedding(
            self.state_dim+1, self.hidden, padding_idx=self.state_dim)
        self.rnncell = nn.GRU(self.hidden, self.hidden,
                              self.num_layers, batch_first=True)

        # self.predict_fc1 = nn.Linear(self.hidden,self.hidden)
        # self.predict_fc2 = nn.Linear(self.hidden,self.action_dim)

        self.activation = torch.relu

    def forward(self, state_seq, seq_len):
        # state_seq = stateseq_in
        # seq_len   = stateseq_len

        x_emb = self.state_emb(state_seq)
        packed_input = pack_padded_sequence(
            x_emb, seq_len.tolist(), batch_first=True)
        h, x_rnn = self.rnncell(packed_input)
        return h, x_rnn


class Value_infoGAIL(Value_vanilla):
    def __init__(self, state_dim, action_dim, encode_dim, hidden: int, num_layers=3):
        super(Value_infoGAIL, self).__init__(state_dim, action_dim, hidden)
        """
        :param name: string
        :param env: gym env
        """
        self.encode_dim = encode_dim
        self.num_layers = num_layers
        self.StateSeqEmb = StateSeqEmb(
            self.state_dim, self.action_dim, self.hidden, num_layers)

        self.encode_fc1 = nn.Linear(
            in_features=self.encode_dim, out_features=self.hidden)
        self.encode_fc2 = nn.Linear(
            in_features=self.hidden, out_features=self.hidden)

        self.fc1 = nn.Linear(in_features=self.hidden*2 +
                             self.action_dim, out_features=self.hidden*2)
        self.fc2 = nn.Linear(in_features=self.hidden*2,
                             out_features=self.hidden*2)
        self.fc3 = nn.Linear(in_features=self.hidden*2,
                             out_features=self.hidden)

    # def forward(self, state):
    def forward(self, state_seq, action, seq_len, encode):
        # state_seq[state_seq == -1] = self.state_dim
        _, x_rnn = self.StateSeqEmb(state_seq, seq_len)
        x_rnn = x_rnn[self.num_layers-1, :, :]

        action = self.one_hotify(action, self.action_dim)

        c = self.activation(self.encode_fc1(encode))
        c = self.activation(self.encode_fc2(c))

        x = torch.cat([x_rnn, action, c], dim=1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        value = self.fc4(x)
        return value


class Policy_infoGAIL(Policy_vanilla):
    def __init__(self, state_dim, action_dim, hidden: int, origins, start_code, env, encode_dim, disttype="categorical", num_layers=3):
        super(Policy_infoGAIL, self).__init__(
            state_dim, action_dim, hidden, origins, start_code, disttype)
        """
        :param name: string
        :param env: gym env
        """
        self.encode_dim = encode_dim
        self.num_layers = num_layers
        self.StateSeqEmb = StateSeqEmb(
            self.state_dim, self.action_dim, self.hidden, num_layers)

        self.encode_fc1 = nn.Linear(
            in_features=self.encode_dim, out_features=self.hidden)
        self.encode_fc2 = nn.Linear(
            in_features=self.hidden, out_features=self.hidden)

        self.fc1 = nn.Linear(in_features=self.hidden*2,
                             out_features=self.hidden*2)
        self.fc2 = nn.Linear(in_features=self.hidden*2,
                             out_features=self.hidden*2)
        self.fc3 = nn.Linear(in_features=self.hidden*2,
                             out_features=self.hidden)

        self.pfc1 = nn.Linear(in_features=self.hidden,
                              out_features=self.hidden)
        self.pfc2 = nn.Linear(in_features=self.hidden,
                              out_features=self.hidden)
        self.pfc3 = nn.Linear(in_features=self.hidden,
                              out_features=self.hidden)

        self.env = env

        self.action_domain = torch.zeros(
            (len(self.env.states), self.env.max_actions)).long()
        for i in range(len(self.env.states)):
            s0 = self.env.states[i]
            if not s0 == self.env.terminal:
                self.action_domain[i, list(self.env.netconfig[s0].keys())] = 1
        self.action_domain = torch.nn.Parameter(
            self.action_domain, requires_grad=False)

    def forward(self, state_seq, seq_len, encode):
        # state_seq = exp_obs
        # seq_len = exp_len
        # self = policy

        # state_seq[state_seq == -1] = self.state_dim
        _, x_rnn = self.StateSeqEmb(state_seq, seq_len)
        x_rnn = x_rnn[self.num_layers-1, :, :]

        c = self.activation(self.encode_fc1(encode))
        c = self.activation(self.encode_fc2(c))

        x = torch.cat([x_rnn, c], axis=1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        last_states = state_seq[torch.arange(state_seq.size(0)), seq_len-1]

        masked_logit = self.origin_fc4(x).masked_fill(
            (1-self.action_domain[last_states]).bool(), -1e32)
        prob = torch.nn.functional.softmax(masked_logit, dim=1)

        # last_states = state_seq[torch.arange(state_seq.size(0)) , seq_len-1]
        # origins_idx = last_states == self.start_code
        # origins_prob = torch.softmax(self.origin_fc4(x[origins_idx]) , dim = 1)
        # others_prob  = torch.softmax(self.fc4(x[~origins_idx]) , dim = 1)

        # prob = torch.zeros(size = (state_seq.shape[0] , self.prob_dim) , device = x.device)
        # prob[origins_idx] = origins_prob
        # prob[~origins_idx,:self.action_dim] = others_prob
        # # action_dist = torch.distributions.Categorical(prob)

        action_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            temperature=1, probs=prob)
        # action_dist = torch.distributions.Categorical(prob)
        return action_dist

    def pretrain_forward(self, state_seq, seq_len):
        h, _ = self.StateSeqEmb(state_seq, seq_len)
        h, seq_len1 = pad_packed_sequence(h, True)

        h = self.activation(self.pfc1(h))
        h = self.activation(self.pfc2(h))
        h = self.activation(self.pfc3(h))

        temp_stateseq = state_seq.view((state_seq.size(0) * state_seq.size(1)))
        temp_h = h.view((state_seq.size(0) * state_seq.size(1), h.size(2)))
        origins_idx = temp_stateseq == self.start_code
        origins_prob = self.origin_fc4(temp_h[origins_idx])
        others_prob = self.fc4(temp_h[~origins_idx])

        prob = torch.zeros(size=(temp_stateseq.size(
            0), self.prob_dim), device=h.device)
        prob[origins_idx] = origins_prob
        prob[~origins_idx, :self.action_dim] = others_prob

        prob = prob.view((state_seq.size(0), state_seq.size(1), self.prob_dim))

        return prob


class Policy_net(Policy_vanilla):
    def __init__(self, state_dim, action_dim, hidden: int, origins, start_code, env, disttype="categorical", num_layers=3):
        super(Policy_net, self).__init__(state_dim, action_dim,
                                         hidden, origins, start_code, env, disttype)
        """
        :param name: string
        :param env: gym env
        """
        self.num_layers = num_layers
        self.StateSeqEmb = StateSeqEmb(
            self.state_dim, self.action_dim, self.hidden, num_layers)
        self.env = env

        # self.action_domain = torch.zeros(
        #     (len(self.env.states), self.env.max_actions)).long()
        # for i in range(len(self.env.states)):
        #     s0 = self.env.states[i]
        #     if not s0 == self.env.terminal:
        #         self.action_domain[i, list(self.env.netconfig[s0].keys())] = 1
        # self.action_domain = torch.nn.Parameter(
        #     self.action_domain, requires_grad=False)

    def forward(self, state_seq, seq_len):
        # state_seq = exp_obs
        # seq_len = exp_len
        # self = policy

        # state_seq[state_seq == -1] = self.state_dim
        _, x_rnn = self.StateSeqEmb(state_seq, seq_len)
        x_rnn = x_rnn[self.num_layers-1, :, :]

        x = self.activation(self.fc1(x_rnn))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        last_states = state_seq[torch.arange(state_seq.size(0)), seq_len-1]

        masked_logit = self.origin_fc4(x).masked_fill(
            (1-self.action_domain[last_states]).bool(), -1e32)
        prob = torch.nn.functional.softmax(masked_logit, dim=1)

        # origins_idx = last_states == self.start_code
        # origins_prob = self.origin_fc4(x[origins_idx]) * self.action_domain[last_states][origins_idx]
        # origins_prob = torch.softmax(origins_prob, dim = 1)

        # origins_prob = self.fc4(x[~origins_idx]) * self.action_domain[last_states][~origins_idx]
        # others_prob  = torch.softmax(self.fc4(x[~origins_idx]) , dim = 1)
        # prob = torch.zeros(size = (state_seq.shape[0] , self.prob_dim) , device = x.device)
        # prob[origins_idx] = origins_prob
        # prob[~origins_idx,:self.action_dim] = others_prob

        action_dist = torch.distributions.Categorical(prob)

        return action_dist

    def pretrain_forward(self, state_seq, seq_len):
        h, _ = self.StateSeqEmb(state_seq, seq_len)
        h, seq_len1 = pad_packed_sequence(h, True)

        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.activation(self.fc3(h))

        temp_stateseq = state_seq.view((state_seq.size(0) * state_seq.size(1)))
        temp_h = h.view((state_seq.size(0) * state_seq.size(1), h.size(2)))
        origins_idx = temp_stateseq == self.start_code
        origins_prob = self.origin_fc4(temp_h[origins_idx])
        others_prob = self.fc4(temp_h[~origins_idx])

        prob = torch.zeros(size=(temp_stateseq.size(
            0), self.prob_dim), device=h.device)
        prob[origins_idx] = origins_prob
        prob[~origins_idx, :self.action_dim] = others_prob

        prob = prob.view((state_seq.size(0), state_seq.size(1), self.prob_dim))

        return prob


class Value_net(Value_vanilla):
    def __init__(self, state_dim, action_dim, hidden: int, num_layers=3):
        super(Value_net, self).__init__(state_dim, action_dim, hidden)
        """
        :param name: string
        :param env: gym env
        """
        self.num_layers = num_layers
        self.StateSeqEmb = StateSeqEmb(
            self.state_dim, self.action_dim, self.hidden, num_layers)

    # def forward(self, state):

    def forward(self, state_seq, action, seq_len):
        # state_seq[state_seq == -1] = self.state_dim
        _, x_rnn = self.StateSeqEmb(state_seq, seq_len)
        x_rnn = x_rnn[self.num_layers-1, :, :]

        action = self.one_hotify(action, self.action_dim)

        x = torch.cat([x_rnn, action], dim=1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        value = self.fc4(x)
        return value
