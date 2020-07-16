import torch
# import tensorflow as tf
import copy
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import datetime


class Trainer:
	def __init__(self,env, Policy,Value,Discrim,args):
		self.env = env
		self.Policy = Policy
		self.Value = Value
		self.Discrim = Discrim

		self.lr = args.learning_rate
		self.eps = args.eps
		self.num_discrim_update = args.num_discrim_update
		self.num_gen_update = args.num_gen_update
		self.gamma = args.gamma
		self.c_1 = args.c_1
		self.c_2 = args.c_2

		self.policy_opt = torch.optim.Adam(self.Policy.parameters(),lr = self.lr,eps=self.eps)
		self.value_opt  = torch.optim.Adam(self.Value.parameters(),lr = self.lr,eps=self.eps)
		self.discrim_opt = torch.optim.Adam(self.Discrim.parameters(),lr = self.lr,eps=self.eps)
		self.pretrain_opt = torch.optim.Adam(self.Policy.parameters(),lr = self.lr,eps=self.eps)

		self.value_criterion = torch.nn.MSELoss()
		self.discrim_criterion = torch.nn.BCELoss()

		now=datetime.datetime.now()
		self.summary = SummaryWriter(logdir = "log/test_{}_lr{}_ud{}_ug{}_g{}_tm{}_data{}_b{}".format(  now.strftime('%Y%m%d_%H%M%S'), 
																						str(self.lr),
																						str(self.num_discrim_update),
																						str(self.num_gen_update),
																						str(self.gamma) ,
																						args.train_mode,
																						args.data,
																						args.batch_size
																						 ))
		self.summary_cnt = 0 
		self.rnn_summary_cnt = 0

	def pretrain_rnn(self, stateseq, seqlen):
		raise NotImplementedError

	def train(self, exp_obs, exp_act , train_mode = "value_policy"):
		raise NotImplementedError
	
	def set_device(self, device):
		self.device =device

	def unroll_trajectory(self, *args, **kwargs):
		raise NotImplementedError