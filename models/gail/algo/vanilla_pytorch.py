import torch
# import tensorflow as tf
import copy
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import datetime
from algo.trainer import Trainer

from models.utils import Step

def atanh(x):
	x = torch.clamp(x, -1+1e-7 , 1-1e-7)
	out = torch.log(1+x) - torch.log(1-x)
	return 0.5*out

class GAILTrain(Trainer):
	def __init__(self,env, Policy,Value,Discrim,args):
		super().__init__(env, Policy, Value, Discrim, args)


	def train_discrim_step(self, exp_obs, exp_act, learner_obs, learner_act):
		expert = self.Discrim(exp_obs, exp_act.detach())
		learner = self.Discrim(learner_obs, learner_act.detach())

		expert_target  = torch.zeros_like(expert)
		learner_target = torch.ones_like(learner)

		discrim_loss = self.discrim_criterion(expert , expert_target) + \
			self.discrim_criterion(learner , learner_target)		

		self.discrim_opt.zero_grad()
		discrim_loss.backward()    
		self.discrim_opt.step()	
		expert_acc  = ((expert < 0.5).float()).mean()
		learner_acc = ((learner > 0.5).float()).mean()

		return expert_acc, learner_acc, discrim_loss


	def calculate_cum_value(self, exp_obs, learner_act):
		rewards = self.Discrim.get_reward(exp_obs, learner_act).squeeze(1) 
		next_obs = torch.zeros_like(exp_obs)
		for i in range(next_obs.shape[0]):
			next_state = self.env.netconfig[self.env.states[exp_obs[i].item()]][learner_act[i].item()]
			next_obs[i] = self.env.states.index(next_state)

		next_act_prob = self.Policy.forward(next_obs)

		action_idxs= torch.Tensor([i for i in range(self.Policy.action_dim)]).long().to(exp_obs.device)
		action_idxs= torch.cat(exp_obs.size(0)*[action_idxs.unsqueeze(0)])
		next_values = torch.cat([self.Value(next_obs, action_idxs[:,i]) for i in [0,1,2]] , dim =1)
		next_value = torch.sum(next_act_prob.probs[:,:self.Policy.action_dim] * next_values, axis = 1)

		cum_value = rewards+ self.gamma * next_value
		return cum_value

	def train(self, exp_obs, exp_act, learner_obs, learner_act, train_mode = "value_policy"):
		self.summary_cnt += 1

		for _ in range(self.num_discrim_update):
			expert_acc, learner_acc, discrim_loss = self.train_discrim_step(exp_obs, exp_act,learner_obs,learner_act)
		
		self.summary.add_scalar('loss/discrim',discrim_loss.item() ,self.summary_cnt )
		self.summary.add_scalar('accuracy/expert',expert_acc.item() ,self.summary_cnt )
		self.summary.add_scalar('accuracy/learner',learner_acc.item() ,self.summary_cnt )
		print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc.item() * 100, learner_acc * 100))

		for _ in range(self.num_gen_update):
			learner_act_prob = self.Policy.forward(learner_obs)
			learner_act = learner_act_prob.sample()			

			if train_mode == "value_policy":
				cum_value = self.calculate_cum_value(learner_obs, learner_act)
			elif train_mode =="only_policy":
				cum_value = self.Discrim.get_reward(learner_obs, learner_act).squeeze(1) 

			loss_policy = (cum_value.detach() * learner_act_prob.log_prob(learner_act) ).mean()

			val_pred = self.Value(learner_obs, learner_act)
			loss_value = self.value_criterion(val_pred , cum_value.detach().view(val_pred.size()))
			
			entropy = learner_act_prob.entropy().mean()

			# construct computation graph for loss
			loss = loss_policy - self.c_1 * loss_value + self.c_2 * entropy
			loss = -loss

			self.policy_opt.zero_grad()
			self.value_opt.zero_grad()
			loss.backward()
			self.policy_opt.step()
			self.value_opt.step()

		self.summary.add_scalar('loss/policy' , loss_policy.item() ,self.summary_cnt )
		self.summary.add_scalar('loss/value'  , loss_value.item()  ,self.summary_cnt )
		self.summary.add_scalar('loss/entropy', entropy.item()     ,self.summary_cnt )
		self.summary.add_scalar('loss/total'  , loss.item()        ,self.summary_cnt )


	def unroll_trajectory(self, *args , **kwargs):
		if kwargs:
			num_trajs = kwargs.get("num_trajs" , 200)
			max_length = kwargs.get("max_length" , 30)
		elif args:
			num_trajs , max_length = args
		else:
			raise Exception("wrong input")

		learner_trajs  = [ ]
		self.env.reset(0)
		state = self.env._cur_state
		for _ in range(num_trajs):
			episode_length = 0
			learner_episode = []
			while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
				if episode_length > max_length:
					break				
				episode_length += 1
				obs = self.env.states.index(state)
				obs = torch.Tensor([obs]).long()
				action = self.Policy.act(obs.to(self.device))
				_, _, next_state, _, done = self.env.step(action)
				action_tensor = torch.tensor([action]).view(obs.size())				
				discrim_reward = self.Discrim.get_reward(state  = obs.to(self.device), 
														action =  action_tensor.to(self.device)  )
				learner_episode.append(Step(cur_state = state,action = action,next_state = next_state,reward = discrim_reward.item(),done = done))
				if done:
					self.env.reset(0)
					state = self.env._cur_state
					break
				else:
					state = next_state
			learner_trajs.append(learner_episode)

		return learner_trajs