


import os 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import torch

# from mdp import gridworld1d
from mdp import value_iteration
from mdp import shortestpath

# from lp_irl import *
from models.utils import *
from models.plotutils import *
from models.maxent_irl import maxent_irl
from models.maxent_irl_stateaction import maxent_irl_stateaction
from models.deep_maxent_irl import deep_maxent_irl, DeepMaxEnt
# from deep_maxent_irl import *


# import sys
# sys.argv = ['-n 1000' , '-d data/Logit.csv' , '-o output']



def main(data0, outpath,lr=0.1 , n_iters=1000 , print_freq=20, gamma=0.5 ):
  LEARNING_RATE = lr
  N_ITERS = n_iters
  PRINT_FREQ = print_freq
  GAMMA = gamma

  dataname = data0.split('/')[-1].split('.csv')[0]

  sw = shortestpath.ShortestPath("data/Network.txt")
  N_STATES = sw.n_states
  trajs = sw.import_demonstrations(data0)
  feat_map = np.eye(N_STATES)

  if not os.path.exists(outpath):
      os.mkdir(outpath)

  # for data0 in dataset:
  dirName = outpath
  if not os.path.exists(dirName):
      os.mkdir(dirName)
      print("Directory " + dirName +  " Created ")
  else:    
      print("Directory " + dirName +  " already exists")
  
  if not os.path.exists(os.path.join(dirName,dataname)):
      os.mkdir(os.path.join(dirName,dataname))
      print("Directory " + os.path.join(dirName,dataname) +  " Created ")
  else:    
      print("Directory " + os.path.join(dirName,dataname) +  " already exists")      

  route_list = identify_routes(trajs)
  # max_route_length = max([x[1] for x in route_list])


  
  ####################
  ### Train MaxEnt ###
  rewards_maxent = maxent_irl(sw,feat_map, GAMMA, trajs, LEARNING_RATE*2, N_ITERS*2 , print_freq=PRINT_FREQ)
  np.save( os.path.join(dirName,dataname,"rewards_maxent.npy"),rewards_maxent)
  
  ########################################
  ### Train MaxEnt State action###
  rewards_maxent_state_action = maxent_irl_stateaction(sw,feat_map, GAMMA, trajs, LEARNING_RATE*2, N_ITERS*2 , print_freq=PRINT_FREQ)
  np.save( os.path.join(dirName,dataname,"rewards_maxent_state_action.npy")  ,rewards_maxent_state_action)

  ########################
  ### Train MaxEnt Deep###
  rewards_deepmaxent,model = deep_maxent_irl(sw,feat_map, GAMMA, trajs, LEARNING_RATE*2, N_ITERS*2 , print_freq=PRINT_FREQ)
  rewards_deepmaxent = rewards_deepmaxent.detach().numpy()
  np.save( os.path.join(dirName,dataname,"rewards_deepmaxent.npy"),rewards_deepmaxent)
  torch.save(model.state_dict(), os.path.join(dirName,dataname,"deep_maxent_model.pytc"))

  rewards_maxent = np.load(os.path.join(dirName,dataname,"rewards_maxent.npy"))
  rewards_maxent_state_action = np.load(os.path.join(dirName,dataname,"rewards_maxent_state_action.npy"))

  _, policy = value_iteration.value_iteration(sw, rewards_maxent, GAMMA, error=0.01)
  generated_maxent = sw.generate_demonstrations(policy , n_trajs = 10000)

  _, policy = value_iteration.action_value_iteration(sw, rewards_maxent_state_action, GAMMA, error=0.01)
  generated_maxent_stateaction = sw.generate_demonstrations(policy , n_trajs = 10000)

  _ , policy = model.action_value_iteration()
  policy = policy.detach().numpy()
  generated_maxent_deep = sw.generate_demonstrations(policy , n_trajs = 10000)

  route_list = identify_routes(trajs)
  routes = [x[0] for x in route_list]

  def check_RouteID(episode,routes):
    state_seq = [str(x.cur_state) for x in episode] + [str(episode[-1].next_state)]
    episode_route = "-".join(state_seq)
    if episode_route in routes:
      idx = routes.index(episode_route)
    else:
      idx = -1
    return idx  

  expert_route = [check_RouteID(x,routes) for x in trajs]
  maxent_route = [check_RouteID(x,routes) for x in generated_maxent]
  maxent_sa_route = [check_RouteID(x,routes) for x in generated_maxent_stateaction]
  maxent_deep_route = [check_RouteID(x,routes) for x in generated_maxent_deep]

  route_idxs= list(range(len(routes)))+[-1]

  route_dist = [(i , expert_route.count(i) , maxent_route.count(i) , maxent_sa_route.count(i), maxent_deep_route.count(i)) for i in route_idxs]
  pd_route_dist = pd.DataFrame(route_dist)
  pd_route_dist.columns = ["routeid",'expert','maxent','maxent_sa','maxent_deep']
  pd_route_dist['expert']    = pd_route_dist['expert']    / sum(pd_route_dist['expert'])
  pd_route_dist['maxent']    = pd_route_dist['maxent']    / sum(pd_route_dist['maxent'])
  pd_route_dist['maxent_sa'] = pd_route_dist['maxent_sa'] / sum(pd_route_dist['maxent_sa'])
  pd_route_dist['maxent_deep'] = pd_route_dist['maxent_deep'] / sum(pd_route_dist['maxent_deep'])
  
  pd_route_dist.to_csv(os.path.join(dirName,dataname,"route_dist.csv"))
  fig = plot_barchart(pd_route_dist, route_idxs, os.path.join(dirName,dataname,"compare.png"), keep_unknown=False)
  fig = plot_seperate_barchart(pd_route_dist, route_idxs, os.path.join(dirName,dataname,"compare.png"), keep_unknown=False)





# if __name__ == "__main__":
#   PARSER = argparse.ArgumentParser()
#   PARSER.add_argument('-n', '--n-iters', default=1000, type=int, help='number of iterations')
#   PARSER.add_argument('-lr', '--learning-rate', default=0.1, type=float, help='learning rate')
#   PARSER.add_argument('-pf', '--print-freq', default=20, type=float, help='print frequency')
#   PARSER.add_argument('-g', '--gamma' , default = 0.5 , type = float , help = 'gamma')
#   PARSER.add_argument('-d' , '--data' , type = str , help = "path of expert demonstration data")
#   PARSER.add_argument('-o' , '--outpath', type= str , help=  "output path")

#   ARGS = PARSER.parse_args()
#   print(ARGS)
#   main(ARGS.data[1:], ARGS.outpath[1:], ARGS.learning_rate, ARGS.n_iters , ARGS.print_freq, ARGS.gamma)
