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


def compute_state_visition_freq(sw, gamma, trajs, policy, deterministic=True):
  """compute the expected states visition frequency p(s| theta, T) 
  using dynamic programming
  inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy
  
  returns:
    p       Nx1 vector - state visitation frequencies
  """
  N_STATES = sw.n_states
  # N_ACTIONS = sw.max_actions

  T = len(trajs[0])+1
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T]) 

  for traj in trajs:
    mu[sw.pos2idx(traj[0].cur_state), 0] += 1
  mu[:,0] = mu[:,0]/len(trajs)

  for t in range(T-1):
    for s in range(N_STATES):
      if deterministic:
        mu[s, t+1] = sum([mu[pre_s, t]*sw.is_connected(sw.idx2pos(pre_s) , np.argmax(policy[pre_s]) , sw.idx2pos(s)) for pre_s in range(N_STATES)])
        # mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, np.argmax(policy[pre_s])] for pre_s in range(N_STATES)])
      else:
        mu_temp = 0
        for pre_s in range(N_STATES):
          action_list = sw.get_action_list(sw.idx2pos(pre_s))
          for a1 in range(len(action_list)):
            mu_temp += mu[pre_s, t]*sw.is_connected(sw.idx2pos(pre_s) , action_list[a1] , sw.idx2pos(s)) *policy[pre_s, a1]

        mu[s, t+1] = mu_temp
        # mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])

  p = np.sum(mu, 1)
  return p



def maxent_irl(sw, feat_map, gamma, trajs, lr, n_iters, print_freq):
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

  # N_STATES = sw.N_STATES
  # N_ACTIONS = sw.N_ACTIONS
  # N_STATES, _, N_ACTIONS = np.shape(P_a)

  # init parameters
  theta = np.random.uniform(size=(feat_map.shape[1],))

  # calc feature expectations
  feat_exp = np.zeros([feat_map.shape[1]])
  for episode in trajs:
    for step in episode:
      feat_exp += feat_map[sw.pos2idx(step.cur_state),:]
    feat_exp += feat_map[sw.pos2idx(step.next_state),:]
  feat_exp = feat_exp/len(trajs)


  # errors = []
  # training
  for iteration in range(n_iters):
    # compute reward function
    rewards = np.dot(feat_map, theta)
    # compute policy
    _, policy = value_iteration.value_iteration(sw, rewards, gamma, error=0.01)
    # compute state visition frequences
    svf = compute_state_visition_freq(sw, gamma, trajs, policy, deterministic=False)
    # compute gradients
    grad = feat_exp - feat_map.T.dot(svf)
    # errors.append(error)
    if iteration % (n_iters/print_freq) == 0:
      error = np.mean(grad**2) **0.5
      print('iteration: {}/{}    error = {}'.format(iteration, n_iters, error)) 
    # update params
    theta += lr * grad 
  rewards = np.dot(feat_map, theta)
  # return sigmoid(normalize(rewards))
  return rewards 
