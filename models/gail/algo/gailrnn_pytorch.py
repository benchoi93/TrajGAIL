import torch
import copy
from tensorboardX import SummaryWriter
import datetime
from algo.trainer import Trainer
from models.utils import Step, WeightClipper
import numpy as np
from models.utils import *
from torch.utils.data import Dataset, DataLoader


def atanh(x):
	x = torch.clamp(x, -1+1e-7 , 1-1e-7)
	out = torch.log(1+x) - torch.log(1-x)
	return 0.5*out


class sequence_data(Dataset):
	def __init__(self, obs, len0, act):
		self.obs = obs
		self.len = len0
		self.act = act

		self.data_size = obs.size(0)

	def __getitem__(self, index):
		return self.obs[index], self.len[index], self.act[index]
	
	def __len__(self):
		return self.data_size
		

class GAILRNNTrain(Trainer):
	def __init__(self,env, Policy,Value,Discrim,pad_idx,args):
		super().__init__(env, Policy, Value, Discrim, args)
		"""
		:param Policy:
		:param Old_Policy:
		:param gamma:
		:param clip_value:x
		:param c_1: parameter for value difference
		:param c_2: parameter for entropy bonus
		"""
		self.pad_idx = pad_idx
		self.num_posterior_update = 2
		self.args = args

		self.find_state = lambda x: env.states.index(x) if x != -1 else pad_idx
		self.nettransition = -1*torch.ones(size = (len(self.env.states) , max(list(map(len , self.env.netconfig.values() ))))).long()
		for key in self.env.netconfig.keys():
			idx = self.find_state(key)
			for key2 in self.env.netconfig[key]:
				self.nettransition[idx,key2] = self.find_state(self.env.netconfig[key][key2])

	def pretrain(self,exp_trajs,find_state,device,args):

		pretrain_trajs = []
		for episode in exp_trajs:
			# pretrain_trajs.append([x.cur_state for x in episode] + [episode[-1].next_state])
			pretrain_trajs.append( [(x.cur_state, x.action )for x in episode]  )
		stateseq_len = np.array([len(x) for x in pretrain_trajs])
		stateseq_maxlen = np.max(stateseq_len)
		stateseq_in = -np.ones((len(pretrain_trajs) , stateseq_maxlen))
		# stateseq_target = -np.ones((len(pretrain_trajs) , stateseq_maxlen-1))
		stateseq_target = -np.ones((len(pretrain_trajs) , stateseq_maxlen))
		
		for i in range(len(pretrain_trajs)):
			temp = pretrain_trajs[i]
			# stateseq_in[i,:len(temp)-1] = temp[:-1]
			# stateseq_target[i,:len(temp)-1] = temp[1:]
			stateseq_in[i, :len(temp)] = [x[0] for x in temp]
			stateseq_target[i, :len(temp)] = [x[1] for x in temp]

		get_num_options = lambda x: len(self.env.netconfig[x]) if x != -1 else 0
		get_num_options = np.vectorize(get_num_options)
		num_options = get_num_options(stateseq_in)

		stateseq_in = find_state(stateseq_in)
		# stateseq_target = find_state(stateseq_target)

		stateseq_in = torch.LongTensor(stateseq_in).to(device)
		stateseq_target = torch.LongTensor(stateseq_target).to(device)
		stateseq_len = torch.LongTensor(stateseq_len).to(device)
		num_options = torch.LongTensor(num_options).to(device)

		stateseq_len , sorted_idx = stateseq_len.sort(0,descending = True)
		stateseq_in = stateseq_in[sorted_idx]
		stateseq_target = stateseq_target[sorted_idx]
		num_options = num_options[sorted_idx]

		out = self.unroll_trajectory2(num_trajs = 200 , max_length = 30)
		learner_observations = out[0]
		learner_actions      = out[1]
		learner_len          = out[2]

		learner_obs = -1 * np.ones((learner_len.sum() , learner_len.max()))
		learner_act = np.zeros((learner_len.sum()))
		learner_l = np.zeros((learner_len.sum()))
		cnt = 0
		for i0 in range(learner_len.shape[0]):
			for j0 in range(1,learner_len[i0]+1):
				try:
					learner_obs[cnt,:j0] = learner_observations[i0,:j0]
					learner_act[cnt] = int(learner_actions[i0][j0-1])
					learner_l[cnt] = j0
					cnt +=1
				except:
					# print("break with index error in Learner Trajectory")
					break
		idx = learner_l !=0
		learner_obs = learner_obs[idx]
		learner_act = learner_act[idx]
		learner_l = learner_l[idx]
		# print(time.time() - now)
		learner_obs, learner_act, learner_len = arr_to_tensor(find_state, device, learner_obs, learner_act, learner_l)

		sample_indices = np.random.randint(low=0, high=len(exp_trajs), size=args.n_episode)
		exp_trajs_temp = np.take(a=exp_trajs, indices=sample_indices, axis=0) 
		exp_obs, exp_act, exp_len = trajs_to_tensor(exp_trajs_temp)
		exp_obs, exp_act, exp_len = arr_to_tensor(find_state, device, exp_obs, exp_act, exp_len)

		expert_dataset = sequence_data(exp_obs, exp_len, exp_act)
		learner_dataset = sequence_data(learner_obs, learner_len, learner_act)

		expert_loader   = DataLoader(dataset=expert_dataset, batch_size=self.args.batch_size, shuffle=True)
		learner_loader = DataLoader(dataset=learner_dataset, batch_size=self.args.batch_size, shuffle=True)




		for i in range(args.pretrain_step):
			acc,acc2,loss = self.pretrain_rnn(stateseq_in , stateseq_target, stateseq_len, num_options)
			for expert_data,learner_data in zip(enumerate(expert_loader) , enumerate(learner_loader)):
				sampled_exp_obs,sampled_exp_len,sampled_exp_act = expert_data[1]
				sampled_exp_len, idxs = torch.sort(sampled_exp_len,descending=True)
				sampled_exp_obs = sampled_exp_obs[idxs]
				sampled_exp_act = sampled_exp_act[idxs]

				sampled_learner_obs,sampled_learner_len,sampled_learner_act = learner_data[1]
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

			if i % 10 == 0:
				print("progress({:.2f})   Acc = {:.5f} // Acc2 = {:.5f} // and Loss = {:.5f}".format(float(i)/float(args.pretrain_step) * 100,acc*100,acc2*100,loss))
				self.summary.add_scalar('Pretrain_gen/_acc', acc ,self.rnn_summary_cnt )            
				self.summary.add_scalar('Pretrain_gen/_acc2', acc2 ,self.rnn_summary_cnt )            
				self.summary.add_scalar('Pretrain_gen/_loss', loss ,self.rnn_summary_cnt )            
				self.summary.add_scalar('Pretrain_discrim/_loss',discrim_loss.item() ,self.rnn_summary_cnt )
				self.summary.add_scalar('Pretrain_discrim/_expert',expert_acc.item() ,self.rnn_summary_cnt )
				self.summary.add_scalar('Pretrain_discrim/_learner',learner_acc.item() ,self.rnn_summary_cnt )
				self.rnn_summary_cnt += 1






	def pretrain_rnn(self, stateseq_in, stateseq_target, stateseq_len, num_options):
		out  = self.Policy.pretrain_forward(stateseq_in, stateseq_len)
		criterion = torch.nn.NLLLoss(ignore_index = self.pad_idx)

		size0 = stateseq_target.size(0) * stateseq_target.size(1)

		out = out.view((size0, self.Policy.prob_dim))
		num_options = num_options.reshape((size0 , ))
		stateseq_target = stateseq_target.view((size0,))

		idx= num_options != 0
		out = out[idx]
		num_options = num_options[idx]
		stateseq_target = stateseq_target[idx]

		mask1= torch.zeros_like(out)
		for l0 in torch.unique(num_options):
			mask1[num_options == l0 , :l0] =1

		out = out + (mask1 + 1e-10).log()
		log_prob = torch.nn.functional.log_softmax(out, dim = 1)

		rnnloss = criterion(input = log_prob,
							target= stateseq_target)
		
		acc = torch.argmax(out,1) == stateseq_target
		acc = acc.float()
		acc = torch.sum(acc) / acc.shape[0]

		prob = torch.softmax(out, 1)
		prob_idx = stateseq_target
		
		idxs = torch.arange(0,prob.shape[0]).to(self.device)
		acc2 = torch.mean(prob[idxs][idxs, prob_idx[idxs]])

		self.pretrain_opt.zero_grad()
		rnnloss.backward()    
		self.pretrain_opt.step()	

		return acc, acc2, rnnloss

	def train_wasser_discrim_step(self, exp_obs, exp_act, exp_len, learner_obs, learner_act,learner_len):

		expert = self.Discrim(exp_obs, exp_act.detach(),exp_len)
		learner = self.Discrim(learner_obs, learner_act.detach(), learner_len)
		# learner_target = -1* torch.ones_like(learner)

		one = torch.FloatTensor([1]).to(expert.device)
		mone = -1*one

		self.discrim_opt.zero_grad()
		dloss_expert = expert.mean(0).view(1)
		# dloss_expert.backward(one)
		dloss_learner = learner.mean(0).view(1)
		# dloss_learner.backward(mone)
		d_loss = -(dloss_expert - dloss_learner)
		d_loss.backward()
		## TODO:: gradient penalty
		self.discrim_opt.step()
		
		weight_clipping = 0.01
		# clipper = WeightClipper()
		# self.Discrim.apply(clipper)
		for p in self.Discrim.parameters():
			p.data.clamp_(-weight_clipping, weight_clipping)		

		self.summary.add_scalar('accuracy/dloss_expert',dloss_expert.detach().item() ,self.summary_cnt )
		self.summary.add_scalar('accuracy/dloss_learner',dloss_learner.detach().item() ,self.summary_cnt )

		expert_acc  = ((expert < 0).float()).mean()
		learner_acc = ((learner > 0).float()).mean()

		return expert_acc, learner_acc, d_loss

	def train_discrim_step(self, exp_obs, exp_act, exp_len, learner_obs, learner_act,learner_len):
		expert = self.Discrim(exp_obs, exp_act.detach(),exp_len)
		learner = self.Discrim(learner_obs, learner_act.detach(), learner_len )

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

	def calculate_cum_value(self, learner_obs, learner_act, learner_len):
		rewards = self.Discrim.get_reward(learner_obs, learner_act , learner_len).squeeze(1) 

		next_obs = self.pad_idx*torch.ones(size = (learner_obs.size(0) , learner_obs.size(1) +1 ) , device = learner_obs.device , dtype = torch.long)
		next_obs[:,:learner_obs.size(1)]=learner_obs
		last_idxs = learner_obs[torch.arange(0,learner_obs.size(0)).long(), learner_len-1]
		next_idxs= self.nettransition[last_idxs, learner_act]
		next_obs[torch.arange(0,learner_obs.size(0)).long() , learner_len] = next_idxs.to(self.device)

		next_len = learner_len+1
		next_obs[next_obs==-1] = self.pad_idx
		next_act_prob = self.Policy.forward(next_obs,next_len)
		
		action_idxs= torch.Tensor([i for i in range(self.Policy.action_dim)]).long().to(learner_obs.device)
		action_idxs= torch.cat(learner_obs.size(0)*[action_idxs.unsqueeze(0)])
		next_values = torch.cat([self.Value(next_obs, action_idxs[:,i] ,  next_len) for i in [0,1,2]] , dim =1)
		next_value = torch.sum(next_act_prob.probs[:,:3] * next_values, axis = 1)

		cum_value = rewards+ self.gamma * next_value
		return cum_value

	def train_policy(self,learner_obs, learner_len, learner_act):
		learner_act_prob = self.Policy.forward(learner_obs, learner_len)
		cum_value = self.calculate_cum_value(learner_obs, learner_act, learner_len)
		
		loss_policy = (cum_value.detach() * learner_act_prob.log_prob(learner_act) ).mean()

		val_pred = self.Value(learner_obs, learner_act , learner_len)
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

		return loss_policy, loss_value, entropy,loss


	def train(self, exp_obs, exp_act , exp_len, learner_obs, learner_act, learner_len, train_mode = "value_policy"):
		self.summary_cnt += 1
		## Train Discriminator

		expert_dataset = sequence_data(exp_obs, exp_len, exp_act)
		learner_dataset = sequence_data(learner_obs, learner_len, learner_act)

		expert_loader   = DataLoader(dataset=expert_dataset, batch_size=self.args.batch_size, shuffle=True)
		learner_loader = DataLoader(dataset=learner_dataset, batch_size=self.args.batch_size, shuffle=True)
		
		for _ in range(self.num_discrim_update):
			result=[]
			for expert_data,learner_data in zip(enumerate(expert_loader) , enumerate(learner_loader)):
				sampled_exp_obs,sampled_exp_len,sampled_exp_act = expert_data[1]
				sampled_exp_len, idxs = torch.sort(sampled_exp_len,descending=True)
				sampled_exp_obs = sampled_exp_obs[idxs]
				sampled_exp_act = sampled_exp_act[idxs]

				sampled_learner_obs,sampled_learner_len,sampled_learner_act = learner_data[1]
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


		# ## Train Posterior
		# for _ in range(self.num_posterior_update):
		# 	loss = self.train_posterior_step(learner_obs, learner_act, learner_len, learner_encode)
			
		## Train Generator
		
		for _ in range(self.num_gen_update):
			result=[]
			for learner_data in  enumerate(learner_loader):
				sampled_learner_obs,sampled_learner_len,sampled_learner_act = learner_data[1]
				sampled_learner_len, idxs = torch.sort(sampled_learner_len,descending=True)
				sampled_learner_obs = sampled_learner_obs[idxs]
				sampled_learner_act = sampled_learner_act[idxs]
				loss_policy, loss_value, entropy, loss = \
					self.train_policy(  sampled_learner_obs.to(self.device), 
										sampled_learner_len.to(self.device), 
										sampled_learner_act.to(self.device)
									)
				result.append([loss_policy.detach(), loss_value.detach(), entropy.detach(),loss.detach()])

		loss_policy = torch.cat([x[0].unsqueeze(0) for x in result],0).mean()
		loss_value  = torch.cat([x[1].unsqueeze(0) for x in result],0).mean()
		entropy     = torch.cat([x[2].unsqueeze(0) for x in result],0).mean()
		loss        = torch.cat([x[3].unsqueeze(0) for x in result],0).mean()

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
				obs = torch.Tensor([self.env.states.index(x.cur_state) for x in learner_episode] + [obs])
				obs.unsqueeze_(0)
				obs = obs.long()
				obs_len = torch.Tensor([obs.shape[1]]).long()
				obs_len = obs_len.long()

				action = self.Policy.act(obs.to(self.device) , 
										 obs_len.to(self.device))

				_, _, next_state, _, done = self.env.step(action)

				action_tensor = torch.tensor([action]).long()

				discrim_reward = self.Discrim.get_reward(state_seq=obs.to(self.device), 
														 action = action_tensor.to(self.device),
													     seq_len = obs_len.to(self.device))
															
				learner_episode.append(Step(cur_state = state,action = action,next_state = next_state,reward = discrim_reward.item(),done = done))
				if done:
					self.env.reset(0)
					state = self.env._cur_state
					break
				else:
					state = next_state
			learner_trajs.append(learner_episode)
		return learner_trajs

		
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

			find_next_state = lambda x : self.env.netconfig[x[0]][x[1]]
			next_obs = np.apply_along_axis(find_next_state , 1 , np.vstack([last_obs,action]).T )
			
			unmasked_next_obs = -1 * np.ones_like(obs_len)
			unmasked_actions = -1 * np.ones_like(obs_len)
			unmasked_reward = -1 * np.ones(obs_len.shape)

			unmasked_next_obs[~done_mask] = next_obs
			unmasked_actions[~done_mask] = action
			unmasked_reward[~done_mask] = reward

			obs = np.c_[obs,unmasked_next_obs]
			actions = np.c_[actions,unmasked_actions]			
			rewards = np.c_[rewards,unmasked_reward]			

			is_done = np.vectorize(lambda x : (x in self.env.destinations) | (x == -1))
			done_mask = is_done(unmasked_next_obs)
			obs_len += 1-done_mask

		actions = actions[:,1:]
		rewards = rewards[:,1:]

		return obs, actions, obs_len.numpy(),rewards

	
