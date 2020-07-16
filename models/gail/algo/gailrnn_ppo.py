import torch
import copy
from tensorboardX import SummaryWriter
import datetime
from algo.trainer import Trainer
from models.utils import Step, WeightClipper
from algo.gailrnn_pytorch import GAILRNNTrain as GAILRNNTrain_vanilla
import numpy as np
from models.utils import *


class GAILRNNTrain(GAILRNNTrain_vanilla):
    def __init__(self,env, Policy,Value,Discrim,pad_idx,args):
        super().__init__(env, Policy,Value,Discrim,pad_idx,args)
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:x
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """
        # self=GAILRNN
        self.args = args
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.clip = args.clip

        find_state = lambda x: self.env.states.index(x) if x != -1 else pad_idx
        self.find_state = np.vectorize(find_state)

    def break_seq(self,observations,length,actions,advants=None,values=None):
        obs = -1 * np.ones((length.sum() , length.max()))
        act = np.zeros((length.sum()))
        l   = np.zeros((length.sum()))

        if advants is not None:
            advants_tmp = -1 * np.ones((length.sum() ))
        if values is not None:
            values_tmp = -1 * np.ones((length.sum() ))
        cnt = 0

        for i0 in range(length.shape[0]):
            for j0 in range(1,length[i0]+1):
                try:
                    obs[cnt,:j0] =  observations[i0,:j0]
                    act[cnt] = int( actions[i0][j0-1])
                    l[cnt] = j0

                    if advants is not None:
                        advants_tmp[cnt] = advants[i0, j0-1]
                    if values is not None:
                        values_tmp[cnt] = values[i0,j0-1]

                    cnt +=1
                except:
                    # print("break with index error in Learner Trajectory")
                    break
        idx =  l !=0
        obs =  obs[idx]
        act =  act[idx]
        l =  l[idx]

        return_list = [obs,  act,  l]
        if advants is not None:
            advants = advants_tmp

        if values is not None:
            values = values_tmp

        return  return_list, advants, values

    def train(self, exp_obs, exp_act , exp_len, learner_obs, learner_act,learner_len,learner_rewards, learner_values):
        """Train GAILRNN with PPO

        Args:
            exp_obs (numpy array): expert observations
            exp_act (numpy array): expert actions
            exp_len (numpy array): expert observations lengths
            learner_obs (numpy array): learner(generated sequence) observations
            learner_act (numpy array): learner(generated sequence) actions
            learner_len (numpy array): learner(generated sequence) observations lengths
            learner_rewards (numpy array): learner(generated sequence) rewards (from Discriminator Network)
            learner_values (numpy array): learner(generated sequence) values (from Value Network)

        """
        learner_returns = []
        learner_advants = []
        for i in range( self.args.n_episode ):
            returns, advants = get_gae(learner_rewards[i],learner_len[i],learner_values[i],self.gamma,self.lamda)
            learner_returns.append(returns.unsqueeze(0))
            learner_advants.append(advants.unsqueeze(0))

        learner_returns = torch.cat(learner_returns , axis = 0)
        learner_advants = torch.cat(learner_advants , axis = 0)

        learner_batch, learner_advants,learner_values = self.break_seq(learner_obs , learner_len , learner_act,learner_advants,learner_values)
        expert_batch, _,_  = self.break_seq(exp_obs , exp_len , exp_act)

        learner_obs,learner_act,learner_len,sorted_idx = arr_to_tensor(self.find_state,self.device, learner_batch[0],learner_batch[1],learner_batch[2])
        exp_obs,exp_act,exp_len,sorted_idx = arr_to_tensor(self.find_state,self.device, expert_batch[0],expert_batch[1],expert_batch[2])

        learner_values = torch.Tensor(learner_values).to(self.device)
        learner_advants = torch.Tensor(learner_advants).to(self.device)

        self.summary_cnt += 1
        ## Train Discriminator
        for _ in range(self.num_discrim_update):
            expert_acc, learner_acc, discrim_loss = self.train_discrim_step(exp_obs, exp_act, exp_len,
                                                                            learner_obs, learner_act, learner_len)
        self.summary.add_scalar('loss/discrim',discrim_loss.item() ,self.summary_cnt )
        self.summary.add_scalar('accuracy/expert',expert_acc.item() ,self.summary_cnt )
        self.summary.add_scalar('accuracy/learner',learner_acc.item() ,self.summary_cnt )
        print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc.item() * 100, learner_acc * 100))
        

        old_dist = self.Policy.forward(learner_obs, learner_len)
        old_policy = old_dist.log_prob(learner_act).detach()
        old_values = self.Value.forward(learner_obs, learner_act, learner_len).detach()
        ## Train Generator
        for _ in range(self.num_gen_update):
            new_dist = self.Policy.forward(learner_obs, learner_len)
            new_policy = new_dist.log_prob(learner_act)
            
            ratio = torch.exp(new_policy - old_policy)
            surrogate = learner_advants * ratio
            # surrogate_loss = surrogate.mean()

            values = self.Value.forward(learner_obs, learner_act, learner_len)
            clipped_values = old_values + torch.clamp(values-old_values , -self.clip, self.clip)

            value_loss1 = self.value_criterion(clipped_values , learner_values.view(clipped_values.shape))
            value_loss2 = self.value_criterion(values         , learner_values.view(values.shape))
            loss_value = torch.max(value_loss1, value_loss2)

            clipped_ratio = torch.clamp(ratio, 1.0-self.clip, 1.0+self.clip)
            clipped_loss = clipped_ratio * learner_advants

            loss_policy = -torch.min(surrogate, clipped_loss).mean()
            entropy = new_dist.entropy().mean()

            loss = loss_policy + self.c_1 * loss_value + self.c_2 * entropy

            self.policy_opt.zero_grad()
            loss.backward( retain_graph = True )
            self.policy_opt.step()

            self.value_opt.zero_grad()
            loss.backward()
            self.value_opt.step()

        self.summary.add_scalar('loss/policy' , loss_policy.item() ,self.summary_cnt )
        self.summary.add_scalar('loss/value'  , loss_value.item()  ,self.summary_cnt )
        self.summary.add_scalar('loss/entropy', entropy.item()     ,self.summary_cnt )
        self.summary.add_scalar('loss/total'  , loss.item()        ,self.summary_cnt )

        
    def unroll_trajectory2(self, *args , **kwargs):
        if kwargs:
            num_trajs = kwargs.get("num_trajs" , 200)
            max_length = kwargs.get("max_length" , 30)
        elif args:
            num_trajs , max_length = args
        else:
            raise Exception("wrong input")


        find_state = lambda x: self.env.states.index(x)
        np_find_state = np.vectorize(find_state)

        obs = np.zeros((num_trajs,1) , int)
        obs_len = np.ones((num_trajs) , np.int64)
        obs_len = torch.LongTensor(obs_len)

        done_mask = np.zeros_like(obs_len , bool)
        actions = np.zeros_like(obs)
        rewards = np.zeros(obs.shape)
        values = np.zeros(obs.shape)


        for i in range(max_length):
            notdone_obs = obs[~done_mask]
            notdone_obslen = obs_len[~done_mask]

            if notdone_obs.shape[0] == 0:
                break

            state = np_find_state(notdone_obs)
            state = torch.LongTensor(state)

            action_dist = self.Policy.forward(state.to(self.device),
                                                notdone_obslen.to(self.device))
            action = action_dist.sample()

            last_state = state.cpu().numpy()[:,-1]
            action = action.cpu().numpy()
            last_obs= np.array(self.env.states)[last_state]

            reward = self.Discrim.get_reward(state.to(self.device), 
                                            torch.LongTensor(action).to(self.device) , 
                                            notdone_obslen.to(self.device)
                                            )
            reward = reward.squeeze(1).cpu().numpy()

            with torch.no_grad():
                value = self.Value( state_seq = state.to(self.device), 
                                    action    = torch.LongTensor(action).to(self.device)  ,
                                    seq_len   = notdone_obslen.to(self.device)
                                    )
            value = value.squeeze(1).cpu().numpy()
            
            find_next_state = lambda x : self.env.netconfig[x[0]][x[1]]
            next_obs = np.apply_along_axis(find_next_state , 1 , np.vstack([last_obs,action]).T )
            
            unmasked_next_obs = -1 * np.ones_like(obs_len)
            unmasked_actions = -1 * np.ones_like(obs_len)
            unmasked_reward = -1 * np.ones(obs_len.shape)
            unmasked_value = -1 * np.ones(obs_len.shape)

            unmasked_next_obs[~done_mask] = next_obs
            unmasked_actions[~done_mask] = action
            unmasked_reward[~done_mask] = reward
            unmasked_value[~done_mask] = value

            obs = np.c_[obs,unmasked_next_obs]
            actions = np.c_[actions,unmasked_actions]			
            rewards = np.c_[rewards,unmasked_reward]			
            values = np.c_[values,unmasked_value]			

            is_done = np.vectorize(lambda x : (x in self.env.destinations) | (x == -1))
            done_mask = is_done(unmasked_next_obs)
            obs_len += 1-done_mask
        
        obs_len = torch.min(obs_len,torch.ones_like(obs_len)*max_length)
        actions = actions[:,1:]
        rewards = rewards[:,1:]
        values = values[:,1:]

        return obs, actions, obs_len.numpy(),rewards,values


