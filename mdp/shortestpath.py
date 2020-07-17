# 1D Gridworld
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pandas as pd
from models.utils.utils import *

class ShortestPath(object):
  """
  
  ShortestPath Algorithm Imitation Learning Environment

  """

  def __init__(self, network_path,origins,destinations):
    self.network_path = network_path
    self.netin = origins
    self.netout = destinations

    try:
      netconfig = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()) , self.network_path) , sep = " " , header = None)
    except:
      netconfig = pd.read_csv( self.network_path , sep = " " , header = None)
      if len(netconfig.columns) <= 3:
        netconfig = pd.read_csv( self.network_path , sep = "," , header = None)

    netconfig = netconfig[[0,1,2]]
    netconfig.columns = ["from","con","to"]
    #1: right-turn 2: straight 3: left-turn

    netconfig_dict = {}
    for i in range(len(netconfig)):
      fromid,con,toid = netconfig.loc[i]

      if fromid in netconfig_dict.keys():
        netconfig_dict[fromid][con-1] = toid
      else:
        netconfig_dict[fromid]={}
        netconfig_dict[fromid][con-1]=toid
    self.netconfig = netconfig_dict
    
    states = list(set(list(netconfig["from"]) + list(netconfig["to"])))
    states += [x for x in destinations if x not in states]
    states += [x for x in origins if x not in states]
    
    self.start = 1000
    self.terminal = 1001
    
    # self.origins = [252, 273, 298, 302, 372, 443, 441, 409, 430, 392, 321, 245 ]
    # self.destinations = [ 253,  276, 301, 299, 376, 447, 442, 400, 420, 393, 322, 246]
    self.origins = states
    self.destinations = states

    self.netconfig[self.start] = {}
    for i in range(len(self.origins)):
      self.netconfig[self.start][i]=self.origins[i]
    for d0 in self.destinations:
      if d0 not in self.netconfig.keys():
        self.netconfig[d0]={}
      self.netconfig[d0][4] = self.terminal

    states = states + [self.start , self.terminal]
    # 0 = start of the trip
    # 1 = end of the trip

    self.states = states
    self.actions = [1,2,3,4]
    #1: right-turn 2: straight 3: left-turn 4:end_trip

    self.n_states = len(self.states)
    self.n_actions = len(self.actions)
    self.max_actions = max([len(self.get_action_list(s)) for s in self.netconfig.keys()])

    self.rewards = [0 for i in range(self.n_states)]

    self.state_action_pair = sum([[(s,a) for a in self.get_action_list(s)] for s in self.states],[])
    self.num_sapair = len(self.state_action_pair)

    self.sapair_idxs=[]
    for i in range(len(self.state_action_pair)):
      state,action = self.state_action_pair[i]
      s = self.pos2idx(state)
      action_list = self.get_action_list(state)
      a = action_list.index(action)
      self.sapair_idxs.append((s,a))

    self.policy_mask = np.zeros([self.n_states, self.max_actions])
    for s,a in self.sapair_idxs:
      self.policy_mask[s,a]=1

    # # for d0 in self.destinations:
    #   # self.rewards[self.pos2idx(d0)] = 1
    
    # self.rewards[self.pos2idx(252)] = 1
    # self.rewards[self.pos2idx(442)] = 1
    # self.rewards[self.pos2idx(1)] = 1



  def pos2idx(self, state):
    """
    input:
      state id
    returns:
      1d index
    """
    return self.states.index(state)

  def idx2pos(self, idx):
    """
    input:
      1d idx
    returns:
      state id
    """
    return self.states[idx]



    


  # def __init__(self, rewards, terminals, move_rand=0.0):
  #   """
  #   inputs:
  #     rewards     1d float array - contains rewards
  #     terminals   a set of all the terminal states
  #   """
  #   self.n_states = len(rewards)
  #   self.rewards = rewards
  #   self.terminals = terminals
  #   self.actions = [-1, 1]
  #   self.n_actions = len(self.actions)
  #   self.move_rand = move_rand


  def get_reward(self, state):
    return self.rewards[self.pos2idx(state)]

  def get_state_transition(self, state, action):
      return self.netconfig[state][action]

  def get_action_list(self, state):
    if state in self.netconfig.keys():
      return list(self.netconfig[state].keys())
    else:
      return list()

  def get_transition_states_and_probs(self, state, action):
    """
    inputs: 
      state       int - state
      action      int - action

    returns
      a list of (state, probability) pair
    """
    return [(self.netconfig[state][action],1)]

  def is_connected(self, state, action ,state1):
    try:
      return self.netconfig[state][action] == state1
    except:
      return False

    # if state == self.start:
    #   return [(self.origins[action] , 1)]
    # elif state in self.destinations:
    #   if action == 2:
    #     return [(self.terminal, 1)]
    #   else:
    #     return None
    # else:
    #   return [(self.netconfig[state][action] , 1)]



  def is_terminal(self, state):
    if state == self.terminal:
    # if state in self.destinations:
      return True
    else:
      return False

  ##############################################
  # Stateful Functions For Model-Free Leanring #
  ##############################################

  def reset(self, start_pos=0):
    self._cur_state = start_pos

  def get_current_state(self):
    return self._cur_state

  def step(self, action):
    """
    Step function for the agent to interact with gridworld
    inputs: 
      action        action taken by the agent
    returns
      current_state current state
      action        input action
      next_state    next_state
      reward        reward on the next state
      is_done       True/False - if the agent is already on the terminal states
    """
    if self.is_terminal(self._cur_state):
      self._is_done = True
      return self._cur_state, action, self._cur_state, self.get_reward(self._cur_state), True

    next_state = self.get_state_transition(self._cur_state , action)
    last_state = self._cur_state

    # st_prob = self.get_transition_states_and_probs(self._cur_state, action)
    # rand_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
    # last_state = self._cur_state
    # next_state = st_prob[rand_idx][0]
    # next_state = self.get_state_transition(self._cur_state , action)

    reward = self.get_reward(last_state)
    self._cur_state = next_state

    if self.is_terminal(next_state):
      self._is_done = True
      return last_state, action, next_state, reward, True
    return last_state, action, next_state, reward, False

  #######################
  # Some util functions #
  #######################

  def get_transition_mat(self):
    """
    get transition dynamics of the gridworld

    return:
      P_a         NxNxN_ACTIONS transition probabilities matrix - 
                    P_a[s0, s1, a] is the transition prob of 
                    landing at state s1 when taking action 
                    a at state s0
    """
    N_STATES = self.n_states
    N_ACTIONS = len(self.actions)
    P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))

    for si in range(N_STATES):
      for a in range(N_ACTIONS):
        # sj = self.get_state_transition(self.idx2pos(si) , self.actions[a])
        # P_a[si,sj,a]  = 1
        
        if not self.idx2pos(si) == self.terminal:
          try:
            probs = self.get_transition_states_and_probs(self.idx2pos(si) , self.actions[a])
            for sj, prob in probs:
              # Prob of si to sj given action a\
              P_a[si, self.pos2idx(sj), a] = prob
          except:
            # 1
            print(str(si) + " " + str(a))
    return P_a


  def generate_demonstrations(self, policy, n_trajs=100, len_traj=20):
    """gatheres expert demonstrations

    inputs:
    gw          Gridworld - the environment
    policy      Nx1 matrix
    n_trajs     int - number of trajectories to generate
    rand_start  bool - randomly picking start position or not
    start_pos   2x1 list - set start position, default [0,0]
    returns:
    trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
    """

    trajs = []
    # right,wrong = 0,0
    cnt = 0
    for i in range(n_trajs):
      try:
        episode = []
        self.reset(self.start)
        cur_state = self.start

        # action_list = list(self.netconfig[self._cur_state].keys())
        # action_prob = policy[self.pos2idx(self._cur_state)]
        # action_idx = np.random.choice(range(len(self.action_list)) , p=action_prob)
        # action = action_list[action_idx]
        # cur_state, action, next_state, reward, is_done = self.step(action)
        # episode.append(Step(cur_state=cur_state, action=action, next_state=next_state, reward=reward, done=is_done))
        # # while not is_done:
        for _ in range(1,len_traj):
          if self.is_terminal(self._cur_state):
            break
          else:
            action_list = list(self.netconfig[self._cur_state].keys())
            action_prob = policy[self.pos2idx(self._cur_state)]
            action_idx = np.random.choice(range(len(action_list)) , p=action_prob[:len(action_list)])
            action = action_list[action_idx]
            cur_state, action, next_state, reward, is_done = self.step(action)
            episode.append(Step(cur_state=cur_state, action=action, next_state=next_state, reward=reward, done=is_done))
        trajs.append(episode)
      except:
        cnt += 1
        print("error count : " + str(cnt) + " / " + str(i))
    return trajs


  def import_demonstrations(self, demopath):
    """gatheres expert demonstrations

    inputs:
    gw          Gridworld - the environment
    policy      Nx1 matrix
    n_trajs     int - number of trajectories to generate
    rand_start  bool - randomly picking start position or not
    start_pos   2x1 list - set start position, default [0,0]
    returns:
    trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
    """

    if demopath.split(".")[1] == "pkl":
      import pickle
      trajs = pickle.load(open(demopath,'rb'))
      route_list = identify_routes(trajs)
      max_route_length = max([x[1] for x in route_list])
      self.max_route_length = max_route_length
      return trajs
      
    elif demopath.split(".")[1] == "csv":
      demo = pd.read_csv(demopath)

      trajs = []
      
      oid_list = list(set(list(demo["oid"])))
      n_trajs = len(oid_list)

      for i in range(n_trajs):
        
        cur_demo = demo[demo["oid"] == oid_list[i]]
        cur_demo = cur_demo.reset_index()

        len_demo = cur_demo.shape[0]
        episode = []

        self.reset(self.start)
        # cur_state = self.start
        # cur_state, action, next_state, reward, is_done = self.step(2)
        # episode.append(Step(cur_state=cur_state, action=self.actions[action], next_state=next_state, reward=reward, done=is_done))
        for i0 in range(len_demo):
          _cur_state = self._cur_state
          _next_state = cur_demo.loc[i0,"sectionId"]

          action_list = self.get_action_list(_cur_state)
          j = [self.get_state_transition(_cur_state , a0) for a0 in action_list].index(_next_state)
          action = action_list[j]

          cur_state, action, next_state, reward, is_done = self.step(action)
          episode.append(Step(cur_state=cur_state, action=action, next_state=next_state, reward=reward, done=is_done))
        cur_state, action, next_state, reward, is_done = self.step(4)
        episode.append(Step(cur_state=cur_state, action=action, next_state=next_state, reward=reward, done=is_done))
        trajs.append(episode)

      route_list = identify_routes(trajs)
      max_route_length = max([x[1] for x in route_list])
      self.max_route_length = max_route_length
      return trajs



