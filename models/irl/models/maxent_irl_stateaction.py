'''
Implementation of maximum entropy inverse reinforcement learning in
  Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning
  https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf

Acknowledgement:
  This implementation is largely influenced by Matthew Alger's maxent implementation here:
  https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py

By Yiren Lu (luyirenmax@gmail.com), May 2017
'''
import numpy as np
import models.irl.algo.value_iteration as value_iteration
# import img_utils
# from models.utils import *

from models.utils.utils import * 


def maxent_irl_stateaction(sw, feat_map, gamma, trajs, lr, n_iters, print_freq):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

  inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                       landing at state s1 when taking action 
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewards
  """

#  gamma = GAMMA
#  lr = 0.1
#  n_iters = 1000
#  print_freq=10

  N_STATES = sw.n_states
  N_ACTIONS = sw.max_actions
  # N_STATES, _, N_ACTIONS = np.shape(P_a)

  # init parameters
  theta = np.random.uniform(size=(N_STATES,N_ACTIONS))
  theta = theta * sw.policy_mask
  # theta = np.random.uniform(size=(feat_map.shape[1],))

  # calc feature expectations
  feat_exp = expert_compute_state_action_visitation_freq(sw,trajs)

  # errors = []
  # training
  for iteration in range(n_iters):
    # compute reward function
    rewards = np.dot(feat_map, theta)
    # compute policy
    # _, _, policy = value_iteration.value_iteration(sw, rewards, gamma, error=0.01)
    _, policy = value_iteration.action_value_iteration(sw, rewards, gamma, error=0.01)
    # compute state visition frequences
    # svf = compute_state_visition_freq(sw, gamma, trajs, policy, deterministic=False)
    svf = compute_state_action_visitation_freq(sw, gamma, trajs, policy, deterministic=False)
    # compute gradients
    grad = feat_exp - feat_map.T.dot(svf)
    # errors.append(error)
    if iteration % (n_iters/print_freq) == 0:
      error = np.mean(np.sum(grad , axis =1)**2) **0.5
      print('iteration: {}/{}    error = {}'.format(iteration, n_iters, error)) 
    # update params
    theta += lr * grad 
  rewards = np.dot(feat_map, theta)
  # return sigmoid(normalize(rewards))
  return rewards 


