import math
import numpy as np

def value_iteration(sw, rewards, gamma, error=0.01):
  """
  static value iteration function. Perhaps the most useful function in this repo
  
  inputs:
    P_a         NxNxN_ACTIONS transition probabilities matrix - 
                              P_a[s0, s1, a] is the transition prob of 
                              landing at state s1 when taking action 
                              a at state s0
    rewards     Nx1 matrix - rewards for all the states
    gamma       float - RL discount
    error       float - threshold for a stop
    deterministic   bool - to return deterministic policy or stochastic policy
  
  returns:
    values    Nx1 matrix - estimated values
    policy    Nx1 (NxN_ACTIONS if non-det) matrix - policy
  """
  # N_STATES, _, N_ACTIONS = np.shape(P_a)
  N_STATES = sw.n_states
  N_ACTIONS = sw.max_actions

  values = np.zeros([N_STATES])
  policy = np.random.uniform(size=([N_STATES , N_ACTIONS]))
  policy = policy * sw.policy_mask  
  # estimate values
  while True:
    while True:
      values_tmp = values.copy()
      for s in range(N_STATES):
        if not sw.is_terminal(sw.idx2pos(s)):
          action_list = sw.get_action_list(sw.idx2pos(s))
          values[s] = max([(rewards[s] + gamma*values_tmp[sw.pos2idx(sw.netconfig[sw.idx2pos(s)][action_list[a]])]) for a in range(len(action_list))])
          # values[s] = max([sum([sw.is_connected(sw.idx2pos(s),action_list[a],sw.idx2pos(s1)) * (rewards[s] + gamma*values_tmp[s1]) for s1 in range(N_STATES)]) for a in range(len(action_list))])
      if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
        break
    
    # generate stochastic policy

    policy_tmp = policy.copy()
    for s in range(N_STATES):
      action_list = sw.get_action_list(sw.idx2pos(s))
      n_possible_actions= len(action_list)
      s1_list =  [sw.netconfig[sw.idx2pos(s)][action_list[a]] for a in range(n_possible_actions)]
      v_s = [values[sw.pos2idx(s1)] for s1 in s1_list]
      v_s = np.exp(v_s)
      if not np.sum(v_s) == 0:
        v_s = np.transpose(v_s/np.sum(v_s))
      v_s = np.vstack([np.array([0 for i in range(n_possible_actions)]) , v_s])
      v_s = np.max(v_s,axis=0)
      if sum(v_s)>1:
        v_s = v_s/np.sum(v_s)
      policy[s,:n_possible_actions] = v_s
    
    if np.max(abs(policy_tmp - policy)) < error**2 : 
      break
  return values, policy

def action_value_iteration(sw, rewards, gamma, error=0.01):
  """
  static value iteration function. Perhaps the most useful function in this repo
  
  inputs:
    sw          Shortestpath World object
    rewards     NxA matrix or Nx1 matrix  - rewards for all the action-state pair or all states
    gamma       float - RL discount
    error       float - threshold for a stop
  
  returns:
    q_table    NxA matrix - estimated values
    policy     NxA matrix - policy
  """
  # N_STATES, _, N_ACTIONS = np.shape(P_a)
  N_STATES = sw.n_states
  N_ACTIONS = sw.max_actions

  if not rewards.shape == (N_STATES, N_ACTIONS):
    if rewards.shape == (N_STATES,):
      rewards = np.tile(rewards , (N_ACTIONS,1)).T
    else:
      raise ValueError("Reward shape Error // expected : {} but given : {}".format(str((N_STATES , N_ACTIONS)),str(rewards.shape)))

  q_table = np.zeros([N_STATES, N_ACTIONS])

  policy = np.random.uniform(size=([N_STATES , N_ACTIONS]))
  policy = policy * sw.policy_mask
  for s in range(N_STATES):
    if not sw.is_terminal(sw.idx2pos(s)):
      policy[s,:] = policy[s,:] / np.sum(policy[s,:])

  while True:
    while True:
      q_table_tmp = q_table.copy()
      for s in range(N_STATES):
        action_list = sw.get_action_list(sw.idx2pos(s))
        for a in range(len(action_list)):
          next_state = sw.netconfig[sw.idx2pos(s)][action_list[a]]
          s1 = sw.pos2idx(next_state)
          if not sw.is_terminal(next_state):
            next_action_list = sw.get_action_list(next_state)
            q_table[s,a] = rewards[s,a] + gamma * sum(policy[s1,:len(next_action_list)] * q_table[s1,:len(next_action_list)])
      # print(np.max(abs(q_table_tmp - q_table)))
      if np.max(abs(q_table_tmp - q_table)) < error :
        break
    
    
    policy_tmp = policy.copy()
    for s in range(N_STATES):
      n_possible_actions= len(sw.get_action_list(sw.idx2pos(s)))
      v_s = q_table[s,:n_possible_actions]
      v_s = np.exp(v_s)
      if not np.sum(v_s) == 0:
        v_s = np.transpose(v_s/np.sum(v_s))
      v_s = np.vstack([np.array([0 for i in range(n_possible_actions)]) , v_s])
      v_s = np.max(v_s,axis=0)
      if sum(v_s)>1:
        v_s = v_s/np.sum(v_s)
      policy[s,:n_possible_actions] = v_s
    
    if np.max(abs(policy_tmp - policy)) < error**2 : 
      break

  return q_table, policy











  # # estimate values
  # while True:
  #   values_tmp = values.copy()
  #   for s in range(N_STATES):
  #     if not sw.is_terminal(sw.idx2pos(s)):
  #       action_list = sw.get_action_list(sw.idx2pos(s))
  #       values[s] = max([sum([sw.is_connected(sw.idx2pos(s),action_list[a],sw.idx2pos(s1)) * (rewards[s] + gamma*values_tmp[s1]) for s1 in range(N_STATES)]) for a in range(len(action_list))])
  #   if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
  #     break

  # policy = np.zeros([N_STATES, N_ACTIONS])
  # for s in range(N_STATES):
  #   n_possible_actions= len(sw.get_action_list(sw.idx2pos(s)))
  #   v_s = q_table[s,:n_possible_actions]
  #   v_s = np.exp(v_s)
  #   if not np.sum(v_s) == 0:
  #     v_s = np.transpose(v_s/np.sum(v_s))
  #   v_s = np.vstack([np.array([0 for i in range(n_possible_actions)]) , v_s])
  #   v_s = np.max(v_s,axis=0)
  #   if sum(v_s)>1:
  #     v_s = v_s/np.sum(v_s)
  #   policy[s,:n_possible_actions] = v_s

  # return values, q_table, policy


# class ValueIterationAgent(object):

#   def __init__(self, mdp, gamma, iterations=100):
#     """
#     The constructor builds a value model from mdp using dynamic programming
    
#     inputs:
#       mdp       markov decision process that is required by value iteration agent definition: 
#                 https://github.com/stormmax/reinforcement_learning/blob/master/envs/mdp.py
#       gamma     discount factor
#     """
#     self.mdp = mdp
#     self.gamma = gamma
#     states = mdp.get_states()
#     # init values
#     self.values = {}

#     for s in states:
#       if mdp.is_terminal(s):
#         self.values[s] = mdp.get_reward(s)
#       else:
#         self.values[s] = 0

#     # estimate values
#     for i in range(iterations):
#       values_tmp = self.values.copy()

#       for s in states:
#         if mdp.is_terminal(s):
#           continue

#         actions = mdp.get_actions(s)
#         v_s = []
#         for a in actions:
#           P_s1sa = mdp.get_transition_states_and_probs(s, a)
#           R_sas1 = [mdp.get_reward(s1) for s1 in [p[0] for p in P_s1sa]]
#           v_s.append(sum([P_s1sa[s1_id][1] * (mdp.get_reward(s) + gamma *
#                                               values_tmp[P_s1sa[s1_id][0]]) for s1_id in range(len(P_s1sa))]))
#         # V(s) = max_{a} \sum_{s'} P(s'| s, a) (R(s,a,s') + \gamma V(s'))
#         self.values[s] = max(v_s)

#   def get_values(self):
#     """
#     returns
#       a dictionary {<state, value>}
#     """
#     return self.values

#   def get_q_values(self, state, action):
#     """
#     returns qvalue of (state, action)
#     """
#     return sum([P_s1_s_a * (self.mdp.get_reward_sas(s, a, s1) + self.gamma * self.values[s1])
#                 for s1, P_s1_s_a in self.mdp.get_transition_states_and_probs(state, action)])

#   def eval_policy_dist(self, policy, iterations=100):
#     """
#     evaluate a policy distribution
#     returns
#       a map {<state, value>}
#     """
#     values = {}
#     states = self.mdp.get_states()
#     for s in states:
#       if self.mdp.is_terminal(s):
#         values[s] = self.mdp.get_reward(s)
#       else:
#         values[s] = 0

#     for i in range(iterations):
#       values_tmp = values.copy()

#       for s in states:
#         if self.mdp.is_terminal(s):
#           continue
#         actions = self.mdp.get_actions(s)
#         # v(s) = \sum_{a\in A} \pi(a|s) (R(s,a,s') + \gamma \sum_{s'\in S}
#         # P(s'| s, a) v(s'))
#         values[s] = sum([policy[s][i][1] * (self.mdp.get_reward(s) + self.gamma * sum([s1_p * values_tmp[s1]
#                                                                                        for s1, s1_p in self.mdp.get_transition_states_and_probs(s, actions[i])]))
#                          for i in range(len(actions))])
#     return values


#   def get_optimal_policy(self):
#     """
#     returns
#       a dictionary {<state, action>}
#     """
#     states = self.mdp.get_states()
#     policy = {}
#     for s in states:
#       policy[s] = [(self.get_action(s), 1)]
#     return policy


#   def get_action_dist(self, state):
#     """
#     args
#       state    current state
#     returns
#       a list of {<action, prob>} pairs representing the action distribution on state
#     """
#     actions = self.mdp.get_actions(state)
#     # \sum_{s'} P(s'|s,a)*(R(s,a,s') + \gamma v(s'))
#     v_a = [sum([s1_p * (self.mdp.get_reward_sas(state, a, s1) + self.gamma * self.values[s1])
#                 for s1, s1_p in self.mdp.get_transition_states_and_probs(state, a)])
#            for a in actions]

#     # I exponentiated the v_s^a's to make them positive
#     v_a = [math.exp(v) for v in v_a]
#     return [(actions[i], v_a[i] / sum(v_a)) for i in range(len(actions))]

#   def get_action(self, state):
#     """
#     args
#       state    current state
#     returns
#       an action to take given the state
#     """
#     actions = self.mdp.get_actions(state)
#     v_s = []
#     for a in actions:
#       P_s1sa = self.mdp.get_transition_states_and_probs(state, a)
#       R_sas1 = [self.mdp.get_reward(s1) for s1 in [p[0] for p in P_s1sa]]
#       v_s.append(sum([P_s1sa[s1_id][1] *
#                       (self.mdp.get_reward(state) +
#                        self.gamma *
#                        self.values[P_s1sa[s1_id][0]]) for s1_id in range(len(P_s1sa))]))
#     a_id = v_s.index(max(v_s))
#     return actions[a_id]







