import os 
import sys

sys.path.append(os.getcwd())

import numpy as np
import argparse
import pandas as pd
import datetime
from tensorboardX import SummaryWriter

from models.irl.algo import value_iteration
from mdp import shortestpath

from models.utils.utils import *
from models.utils.plotutils import *
from models.irl.models.maxent_irl import maxent_irl
from models.irl.models.maxent_irl_stateaction import maxent_irl_stateaction


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr','--learning-rate', default=0.1, type=float)
    parser.add_argument('-g','--gamma', default=0.5, type=float)
    parser.add_argument('-n','--n-iters',default=int(1000),type =int)
    parser.add_argument('-pf','--print-freq',default=int(20),type =int)
    parser.add_argument('-d' , '--data', default = "data/Single_OD/Binomial.csv")
    return parser.parse_args()
args = argparser()

# def main(data0, outpath,lr=0.1 , n_iters=1000 , print_freq=20, gamma=0.5 ):
LEARNING_RATE = args.learning_rate
N_ITERS = args.n_iters
PRINT_FREQ = args.print_freq
GAMMA = args.gamma
data0 = args.data

dataname = data0.split('/')[-1].split('.csv')[0]

origins = [252, 273, 298, 302, 372, 443, 441, 409, 430, 392, 321, 245 ]
destinations = [ 253,  276, 301, 299, 376, 447, 442, 400, 420, 393, 322, 246]

sw = shortestpath.ShortestPath("data/Network.txt",origins, destinations)
N_STATES = sw.n_states
trajs = sw.import_demonstrations(data0)
feat_map = np.eye(N_STATES)

MaxEnt_svf = model_summary_writer("test_MaxEnt_svf",sw)
MaxEnt_savf = model_summary_writer("test_MaxEnt_savf",sw)

dirName = "Result"
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

####################
### Train MaxEnt ###
rewards_maxent = maxent_irl(sw,feat_map, GAMMA, trajs, LEARNING_RATE*2, N_ITERS*2 , print_freq=PRINT_FREQ)
np.save( os.path.join(dirName,dataname,"rewards_maxent.npy"),rewards_maxent)

########################################
### Train MaxEnt State action###
rewards_maxent_state_action = maxent_irl_stateaction(sw,feat_map, GAMMA, trajs, LEARNING_RATE*2, N_ITERS*2 , print_freq=PRINT_FREQ)
np.save( os.path.join(dirName,dataname,"rewards_maxent_state_action.npy")  ,rewards_maxent_state_action)

rewards_maxent = np.load(os.path.join(dirName,dataname,"rewards_maxent.npy"))
rewards_maxent_state_action = np.load(os.path.join(dirName,dataname,"rewards_maxent_state_action.npy"))

_, policy = value_iteration.value_iteration(sw, rewards_maxent, GAMMA, error=0.01)
generated_maxent = sw.generate_demonstrations(policy , n_trajs = 10000)

_, policy = value_iteration.action_value_iteration(sw, rewards_maxent_state_action, GAMMA, error=0.01)
generated_maxent_stateaction = sw.generate_demonstrations(policy , n_trajs = 10000)

plot_summary_maxent(MaxEnt_svf, trajs , generated_maxent)
plot_summary_maxent(MaxEnt_savf, trajs , generated_maxent_stateaction)

