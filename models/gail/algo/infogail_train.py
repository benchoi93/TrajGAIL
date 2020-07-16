import time
import torch
import copy
from tensorboardX import SummaryWriter
import datetime
from algo.trainer import Trainer
from models.utils import Step, WeightClipper
import numpy as np

from algo.gailrnn_pytorch import GAILRNNTrain
from torch.utils.data import Dataset, DataLoader


class sequence_data(Dataset):
    def __init__(self, type, obs, len0, act, encode=None, act_prob=None):
        self.type = type
        self.obs = obs
        self.len = len0
        self.act = act
        self.encode = encode
        self.act_prob = act_prob
        self.data_size = obs.size(0)

    def __getitem__(self, index):
        if self.type == "learner":
            out = [self.obs[index], self.len[index], self.act[index] , self.encode[index] , self.act_prob[index]]
        elif self.type == "expert":
            out = [self.obs[index], self.len[index], self.act[index] ]
        return out

    def __len__(self):
        return self.data_size
        


class INFOGAILTrain(GAILRNNTrain):
    def __init__(self,env, Policy,Value,Discrim,Posteriors, pad_idx,args):
        super().__init__(env, Policy,Value,Discrim,pad_idx,args)

        self.Posteriors = Posteriors
        self.posterior_criterion = torch.nn.CrossEntropyLoss()
        self.posterior_opt = torch.optim.Adam( sum([list(qnet.parameters()) for qnet in self.Posteriors] , []) , lr = self.args.learning_rate , eps = self.args.eps)
        self.posterior_criterion = torch.nn.CrossEntropyLoss()

        # self.posterior_opt = []
        # for qnet in self.Posteriors:
        #     self.posterior_opt.append(torch.optim.Adam(qnet.parameters(),lr = 0.1,eps=self.eps))

    def train_posterior_step(self, learner_obs, learner_act, learner_len, learner_encode, learner_act_prob):

        encode1 = learner_encode[:,:12]
        encode2 = learner_encode[:,12:]

        # learner_act_prob = self.one_hotify(learner_act, self.Policy.origin_dim)

        c_est1 = self.Posteriors[0](learner_obs, learner_act_prob, learner_len)
        c_est2 = self.Posteriors[1](learner_obs, learner_act_prob, learner_len)
        
        loss_q1 = self.posterior_criterion(c_est1, torch.argmax(encode1,1))
        loss_q2 = self.posterior_criterion(c_est2, torch.argmax(encode2,1))

        return loss_q1 , loss_q2


    def one_hotify(self, longtensor, dim):
        if self.device.type == "cuda":
            one_hot = torch.cuda.FloatTensor(longtensor.size(0) , dim).to()
        else:
            one_hot = torch.FloatTensor(longtensor.size(0) , dim).to()
        one_hot.zero_()
        one_hot.scatter_(1,longtensor.unsqueeze(1).long(),1)
        return one_hot


    def calculate_cum_value(self, learner_obs, learner_act, learner_len,learner_encode):
        rewards = self.Discrim.get_reward(learner_obs, learner_act , learner_len).squeeze(1) 

        next_obs = self.pad_idx*torch.ones(size = (learner_obs.size(0) , learner_obs.size(1) +1 ) , device = learner_obs.device , dtype = torch.long)
        next_obs[:,:learner_obs.size(1)]=learner_obs
        last_idxs = learner_obs[torch.arange(0,learner_obs.size(0)).long(), learner_len-1]
        next_idxs= self.nettransition[last_idxs, learner_act]
        next_obs[torch.arange(0,learner_obs.size(0)).long() , learner_len] = next_idxs.to(self.device)

        next_len = learner_len+1
        next_obs[next_obs==-1] = self.pad_idx
        next_act_prob = self.Policy.forward(next_obs,next_len,learner_encode)
        
        action_idxs= torch.Tensor([i for i in range(self.Policy.action_dim)]).long().to(learner_obs.device)
        action_idxs= torch.cat(learner_obs.size(0)*[action_idxs.unsqueeze(0)])
        next_values = torch.cat([self.Value(next_obs, action_idxs[:,i] ,  next_len,learner_encode) for i in [0,1,2]] , dim =1)
        next_value = torch.sum(next_act_prob.probs[:,:3] * next_values, axis = 1)

        cum_value = rewards+ self.gamma * next_value
        return cum_value


    def train_policy(self,learner_obs, learner_len, learner_act, learner_encode,learner_act_prob):
        learner_act_dist = self.Policy.forward(learner_obs, learner_len, learner_encode).probs
        learner_act_dist = torch.distributions.Categorical(learner_act_dist)

        cum_value = self.calculate_cum_value(learner_obs, learner_act, learner_len,learner_encode)
        loss_policy = (cum_value.detach() * learner_act_dist.log_prob(learner_act) ).mean()

        val_pred = self.Value(learner_obs, learner_act , learner_len,learner_encode)
        loss_value = self.value_criterion(val_pred , cum_value.detach().view(val_pred.size()))
        entropy = learner_act_dist.entropy().mean()
        # construct computation graph for loss
        loss = loss_policy - self.c_1 * loss_value + self.c_2 * entropy
        loss = -loss

        loss_q1, loss_q2 = self.train_posterior_step(learner_obs, learner_act, learner_len, learner_encode,learner_act_prob)
        loss_posterior = loss_q1 + loss_q2
        posterior_entropy = 0
        posterior_entropy += torch.distributions.Categorical(self.Posteriors[0].encode_dist.probs).entropy()
        posterior_entropy += torch.distributions.Categorical(self.Posteriors[1].encode_dist.probs).entropy()
        posterior = loss_posterior - posterior_entropy


        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        self.posterior_opt.zero_grad()

        loss.backward(retain_graph=True)
        posterior.backward(retain_graph=True)

        self.policy_opt.step()
        self.value_opt.step()
        self.posterior_opt.step()
        return loss_policy, loss_value, entropy, loss, loss_q1, loss_q2


    def train(self, \
                exp_obs    , exp_act    , exp_len, \
                learner_obs, learner_act, learner_len, learner_encode, learner_act_prob, \
                train_mode = "value_policy"):
                
        self.summary_cnt += 1
        expert_dataset = sequence_data("expert",exp_obs, exp_len, exp_act)
        learner_dataset = sequence_data("learner",learner_obs, learner_len, learner_act, learner_encode,learner_act_prob)

        expert_loader   = DataLoader(dataset=expert_dataset, batch_size=self.args.batch_size, shuffle=True)
        learner_loader  = DataLoader(dataset=learner_dataset, batch_size=self.args.batch_size, shuffle=True)


        ## Train Discriminator
        for _ in range(self.num_discrim_update):
            result=[]
            for expert_data,learner_data in zip(enumerate(expert_loader) , enumerate(learner_loader)):
                sampled_exp_obs,sampled_exp_len,sampled_exp_act = expert_data[1]
                sampled_exp_len, idxs = torch.sort(sampled_exp_len,descending=True)
                sampled_exp_obs = sampled_exp_obs[idxs]
                sampled_exp_act = sampled_exp_act[idxs]

                sampled_learner_obs,sampled_learner_len,sampled_learner_act,_,_ = learner_data[1]
                sampled_learner_len, idxs = torch.sort(sampled_learner_len,descending=True)
                sampled_learner_obs = sampled_learner_obs[idxs]
                sampled_learner_act = sampled_learner_act[idxs]


                expert_acc, learner_acc, discrim_loss = \
                    self.train_discrim_step(sampled_exp_obs.to(self.device), 
                                            sampled_exp_act.to(self.device), 
                                            sampled_exp_len.to(self.device), 
                                            sampled_learner_obs.to(self.device), 
                                            sampled_learner_act.to(self.device), 
                                            sampled_learner_len.to(self.device)
                                            )

                result.append([expert_acc.detach(), learner_acc.detach(), discrim_loss.detach()])

        dloss = torch.cat([x[2].unsqueeze(0) for x in result],0).mean()
        e_acc = torch.cat([x[0].unsqueeze(0) for x in result],0).mean()
        l_acc = torch.cat([x[1].unsqueeze(0) for x in result],0).mean()

        self.summary.add_scalar('loss/discrim'     ,dloss.item() ,self.summary_cnt )
        self.summary.add_scalar('accuracy/expert'  ,e_acc.item() ,self.summary_cnt )
        self.summary.add_scalar('accuracy/learner' ,l_acc.item() ,self.summary_cnt )
        print("Expert: %.2f%% | Learner: %.2f%%" % (e_acc * 100, l_acc * 100))


        ## Train Generator & Posterior
        for _ in range(self.num_gen_update):
            result=[]
            for learner_data in  enumerate(learner_loader):
                sample = learner_data[1]

                sampled_learner_obs = sample[0]
                sampled_learner_len = sample[1]
                sampled_learner_act = sample[2]
                sampled_learner_encode = sample[3]
                sampled_learner_act_prob  = sample[4]

                sampled_learner_len, idxs = torch.sort(sampled_learner_len,descending=True)
                sampled_learner_obs = sampled_learner_obs[idxs]
                sampled_learner_act = sampled_learner_act[idxs]
                sampled_learner_encode =sampled_learner_encode[idxs]

                loss_policy, loss_value, entropy, loss, loss_q1, loss_q2 = \
                    self.train_policy( learner_obs =  sampled_learner_obs.to(self.device),
                                       learner_len =  sampled_learner_len.to(self.device),
                                       learner_act = sampled_learner_act.to(self.device),
                                       learner_encode = sampled_learner_encode.to(self.device),
                                       learner_act_prob = sampled_learner_act_prob.to(self.device)
                                    )

                result.append([loss_policy.detach(), loss_value.detach(),  entropy.detach(),\
                                loss.detach(),  loss_q1.detach(),  loss_q2.detach()])

        loss_policy = torch.cat([x[0].unsqueeze(0) for x in result],0).mean()
        loss_value  = torch.cat([x[1].unsqueeze(0) for x in result],0).mean()
        entropy     = torch.cat([x[2].unsqueeze(0) for x in result],0).mean()
        loss        = torch.cat([x[3].unsqueeze(0) for x in result],0).mean()
        lossq1      = torch.cat([x[4].unsqueeze(0) for x in result],0).mean()
        lossq2      = torch.cat([x[5].unsqueeze(0) for x in result],0).mean()

        self.summary.add_scalar('loss/policy' , loss_policy.item() ,self.summary_cnt )
        self.summary.add_scalar('loss/value'  , loss_value.item()  ,self.summary_cnt )
        self.summary.add_scalar('loss/entropy', entropy.item()     ,self.summary_cnt )
        self.summary.add_scalar('loss/total'  , loss.item()        ,self.summary_cnt )
        self.summary.add_scalar('loss/q1'  , lossq1.item()        ,self.summary_cnt )
        self.summary.add_scalar('loss/q2'  , lossq2.item()        ,self.summary_cnt )






    def unroll_trajectory2(self, *args , **kwargs):
        if kwargs:
            num_trajs = kwargs.get("num_trajs" , 200)
            max_length = kwargs.get("max_length" , 30)
        elif args:
            num_trajs , max_length = args
        else:
            raise Exception("wrong input")
        
        # self=GAILRNN
        # num_trajs=2000
        # max_length=30

        encodes = []
        for p0 in self.Posteriors:
            encodes.append(p0.encode(num_trajs))
        encode = torch.cat(encodes, axis = 1)


        find_state = lambda x: self.env.states.index(x)
        np_find_state = np.vectorize(find_state)
        obs = np.zeros((num_trajs,1) , int)
        obs_len = np.ones((num_trajs) , np.int64)
        obs_len = torch.LongTensor(obs_len)
        done_mask = np.zeros_like(obs_len , bool)
        actions = np.zeros_like(obs)
        rewards = np.zeros(obs.shape)

        action_probs = torch.zeros((num_trajs , max(len(self.env.actions) , len(self.env.origins)))).unsqueeze(1)
        action_probs = action_probs.to(self.device)


        for i in range(max_length):

            notdone_obs = obs[~done_mask]
            notdone_obslen = obs_len[~done_mask]
            notdone_encode = encode[~done_mask]

            if notdone_obs.shape[0] == 0:
                break

            state = np_find_state(notdone_obs)
            state = torch.LongTensor(state)

            action_dist = self.Policy.forward(state.to(self.device),
                                                notdone_obslen.to(self.device),
                                                notdone_encode.to(self.device))
            action_prob = action_dist.rsample()

            last_state = state.cpu().numpy()[:,-1]
            temp = action_prob * (self.nettransition[last_state] != -1).to(self.device)
            action = torch.argmax(temp, 1)
            action = action.cpu().numpy()
            last_obs = np.array(self.env.states)[last_state]

            reward = self.Discrim.get_reward(state.to(self.device), 
                                            torch.LongTensor(action).to(self.device) , 
                                            notdone_obslen.to(self.device)
                                            )
            reward = reward.squeeze(1).cpu().numpy()
            
            next_state = self.nettransition[last_state , action]
            next_obs = np.array(self.env.states)[next_state]

            unmasked_next_obs = -1 * np.ones_like(obs_len)
            unmasked_actions = -1 * np.ones_like(obs_len)
            unmasked_reward = -1 * np.ones(obs_len.shape)
            unmasked_action_prob = -1 * torch.ones((action_probs.size(0),1,action_probs.size(2))).to(self.device)
            
            unmasked_next_obs[~done_mask] = next_obs
            unmasked_actions[~done_mask] = action
            unmasked_reward[~done_mask] = reward
            unmasked_action_prob[~done_mask,0,:] = action_prob

            obs = np.c_[obs,unmasked_next_obs]
            actions = np.c_[actions,unmasked_actions]			
            rewards = np.c_[rewards,unmasked_reward]			
            action_probs = torch.cat([action_probs , unmasked_action_prob], dim =1)

            is_done = np.vectorize(lambda x : (x in self.env.destinations) | (x == -1))
            done_mask = is_done(unmasked_next_obs)
            obs_len += 1-done_mask

        actions = actions[:,1:]
        rewards = rewards[:,1:]
        action_probs = action_probs[:,1:,:]

        return obs, actions, obs_len.numpy(), rewards, encode, action_probs

