import torch
# import tensorflow as tf
import copy
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import datetime


def atanh(x):
	x = torch.clamp(x, -1+1e-7 , 1-1e-7)
	out = torch.log(1+x) - torch.log(1-x)
	return 0.5*out

class PPOTrain:
	def __init__(self, Policy, Old_Policy, Value,gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01,lr=5e-5,eps=1e-4):
		"""
		:param Policy:
		:param Old_Policy:
		:param gamma:
		:param clip_value:x
		:param c_1: parameter for value difference
		:param c_2: parameter for entropy bonus
		"""

		self.Policy = Policy
		self.Old_Policy = Old_Policy
		self.Value = Value
		self.gamma = gamma
		self.clip_value = clip_value
		self.c_1 = c_1
		self.c_2 = c_2
		self.lr = lr
		self.eps = eps

		self.policy_opt = torch.optim.Adam(self.Policy.parameters(),lr = self.lr,eps=self.eps)
		self.value_opt  = torch.optim.Adam(self.Value.parameters(),lr = self.lr,eps=self.eps)

		now=datetime.datetime.now()
		self.summary = SummaryWriter(logdir = "log/test_{}_lr{}_act{}".format(now.strftime('%Y%m%d_%H%M%S'),str(lr),str(Policy.action_dim)))
		self.summary_cnt = 0 

# self = PPOTrain(policy,old_policy)
	def train(self,  obs, actions, rewards, v_preds_next, gaes,policy_outs=None):
		new_dist = self.Policy.forward(obs)
		old_dist = self.Old_Policy.forward(obs)
		
		log_action_probs		= new_dist.log_prob(actions)
		old_log_action_probs	= old_dist.log_prob(actions)

		ratio = torch.exp( log_action_probs - old_log_action_probs )
		clipped_ratio = torch.clamp(ratio , min=1-self.clip_value , max = 1+self.clip_value)
		loss_clip = torch.min((gaes * ratio) , (gaes * clipped_ratio) )
		loss_policy = loss_clip.mean()
		self.summary.add_scalar('loss/policy', loss_policy.item() ,self.summary_cnt )

		entropy = new_dist.entropy().mean()
		self.summary.add_scalar('loss/entropy', entropy.item() ,self.summary_cnt )

		v_preds = self.Value.forward(obs)
		target_v_preds = rewards + self.gamma * v_preds_next
		target_v_preds = target_v_preds.view(size=v_preds.size()).detach()
		loss_value = torch.nn.functional.mse_loss(v_preds , target_v_preds)
		self.summary.add_scalar('loss/value', loss_value.item() ,self.summary_cnt )

		# construct computation graph for loss
		loss = loss_policy - self.c_1 * loss_value + self.c_2 * entropy
		self.summary.add_scalar('loss/total', loss.item() ,self.summary_cnt )

		# minimize -loss == maximize loss
		loss = -loss

		if loss != loss:
			raise ValueError
			print("Nan!!!!")


		self.policy_opt.zero_grad()
		self.value_opt.zero_grad()
		loss.backward()
		self.policy_opt.step()
		self.policy_opt.step()

		self.summary_cnt += 1

	def train_trajs(self,  trajs, device):
		for i in range(len(trajs)):
			obs, _, reward, v_pred,v_pred_next,_ = trajs[i]

			obs = torch.cat(obs).float().to(device)
			rewards = torch.Tensor(reward).to(device)
			v_preds_next = torch.Tensor(v_pred_next).to(device)

			new_dist = self.Policy.forward(obs)
			old_dist = self.Old_Policy.forward(obs)

			samples = new_dist.rsample()
			actions = torch.tanh(samples)
			gaes = self.get_gaes(rewards=rewards, v_preds=v_pred, v_preds_next=v_pred_next)
			gaes = torch.Tensor(gaes).to(device)

			log_action_probs		= new_dist.log_prob(samples) - torch.log(torch.clamp(1 - actions.pow(2) , 1, 1e-10 ))
			old_log_action_probs	= old_dist.log_prob(samples) - torch.log(torch.clamp(1 - actions.pow(2) , 1, 1e-10 ))

			ratio = torch.exp( log_action_probs - old_log_action_probs )
			clipped_ratio = torch.clamp(ratio , min=1-self.clip_value , max = 1+self.clip_value)
			loss_clip = torch.min((gaes * ratio) , (gaes * clipped_ratio) )
			loss_policy = loss_clip.mean()
			self.summary.add_scalar('loss/policy', loss_policy.item() ,self.summary_cnt )

			entropy = new_dist.entropy().mean()
			self.summary.add_scalar('loss/entropy', entropy.item() ,self.summary_cnt )

			v_preds = self.Value.forward(obs)
			target_v_preds = rewards + self.gamma * v_preds_next
			target_v_preds = target_v_preds.view(size=v_preds.size()).detach()
			loss_value = torch.nn.functional.mse_loss(v_preds , target_v_preds)
			self.summary.add_scalar('loss/value', loss_value.item() ,self.summary_cnt )

			# construct computation graph for loss
			loss = loss_policy - self.c_1 * loss_value + self.c_2 * entropy
			self.summary.add_scalar('loss/total', loss.item() ,self.summary_cnt )

			# minimize -loss == maximize loss
			loss = -loss

			if loss != loss:
				print(1)

			self.policy_opt.zero_grad()
			self.value_opt.zero_grad()
			loss.backward()
			self.policy_opt.step()
			self.policy_opt.step()

			self.summary_cnt += 1


	def hard_update(self,target, source):
		"""
		Copies the parameters from source network to target network
		:param target: Target network (PyTorch)
		:param source: Source network (PyTorch)
		:return:
		"""
		for target_param, param in zip(target.parameters(), source.parameters()):
				target_param.data.copy_(param.data)

	def get_gaes(self, rewards, v_preds, v_preds_next):
		deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
		# calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
		gaes = copy.deepcopy(deltas)
		for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
			gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
		return gaes
