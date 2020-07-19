import numpy as np
import torch

class MMC_predictor(object):
    def __init__(self, sw, max_length):
        self.sw = sw
        self.pad_idx = len(sw.states) + 1
        self.find_idx = lambda x : sw.states.index(x)

    def train(self, trajs):
        transition = np.zeros((len(self.sw.states),len(self.sw.states)))
        #training
        for episode in trajs:
            for step in episode:
                from_idx = self.find_idx(step.cur_state)
                to_idx   = self.find_idx(step.next_state)

                transition[from_idx, to_idx] += 1

        for from_idx in range(transition.shape[0]):
            if np.sum(transition[from_idx , :]) != 0:
                transition[from_idx , :] = transition[from_idx , :] / np.sum(transition[from_idx , :])
            else:
                pass

        transition = torch.Tensor(transition)
        return transition
    
    def unroll_trajectories(self,transition, num_trajs, max_length):
        start_state = self.sw.start
        end_state = self.sw.terminal

        end_idx = self.find_idx(end_state)
        learner_trajs  = torch.ones((num_trajs, 1)).long() * self.find_idx(start_state)
        done_mask = torch.zeros((num_trajs)).bool()

        for i in range(max_length):
            input_trajs = learner_trajs[~done_mask,-1]

            next_prob = transition[input_trajs,:]
            next_prob_dist = torch.distributions.Categorical(next_prob)
            next_idx = next_prob_dist.sample()
            next_idx.unsqueeze_(1)
            
            next_idx_whole = torch.ones((num_trajs, 1)).long() * self.pad_idx
            next_idx_whole[~done_mask] = next_idx

            learner_trajs = torch.cat([learner_trajs, next_idx_whole],dim=1)       
            is_done = (next_idx_whole == end_idx) | (next_idx_whole == self.pad_idx)

            done_mask = done_mask | is_done.view(done_mask.size())

        return learner_trajs

