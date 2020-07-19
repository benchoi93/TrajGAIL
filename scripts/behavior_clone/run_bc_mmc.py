import os
import sys

# os.chdir('/Users/seongjinchoi/Downloads/TrajGen_GAIL')
# sys.argv=['']

sys.path.append(os.getcwd())

import numpy as np
import argparse
import pandas as pd
import datetime
from tensorboardX import SummaryWriter

from mdp import shortestpath

from models.utils.utils import *
from models.utils.plotutils import *

from models.behavior_clone.mmc_predictor import *

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr','--learning-rate', default=0.1, type=float)
    parser.add_argument('-g','--gamma', default=0.5, type=float)
    parser.add_argument('-n','--n-iters',default=int(1000),type =int)
    parser.add_argument('-hd','--hidden',default=int(256),type =int)
    parser.add_argument('-pf','--print-freq',default=int(20),type =int)
    parser.add_argument('-d' , '--data', default = "data/Single_OD/Binomial.csv")
    parser.add_argument('--gangnam', default = False, action = "store_true" )
    return parser.parse_args()
args = argparser()

# def main(data0, outpath,lr=0.1 , n_iters=1000 , print_freq=20, gamma=0.5 ):
LEARNING_RATE = args.learning_rate
N_ITERS = args.n_iters
PRINT_FREQ = args.print_freq
GAMMA = args.gamma
data0 = args.data

dataname = data0.split('/')[-1].split('.csv')[0]
demand_type = data0.split('/')[-2]

if args.gangnam:
    origins=[222,223,224,225,226,227,228,
            214,213,212,211,210,209,208,
            190,189,188,187,186,185,184,
            167,168,169,170,171,172,173,174,175,176]

    destinations=[191,192,193,194,195,196,197,
                183,182,181,180,179,178,177,
                221,220,219,218,217,216,215,
                198,199,200,201,202,203,204,205,206,207 ]
    sw = shortestpath.ShortestPath("data/gangnam_Network.txt",origins,destinations)
else:
    origins = [252, 273, 298, 302, 372, 443, 441, 409, 430, 392, 321, 245 ]
    destinations = [ 253,  276, 301, 299, 376, 447, 442, 400, 420, 393, 322, 246]
    sw = shortestpath.ShortestPath("data/Network.txt",origins, destinations)


N_STATES = sw.n_states
trajs = sw.import_demonstrations(data0)
feat_map = np.eye(N_STATES)

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


BC_MMC = model_summary_writer("test_BC_MMC_{}_{}".format(demand_type, dataname),sw)
max_length = max(list(map(len , trajs)))

MMCMODEL = MMC_predictor(sw, max_length)

transition = MMCMODEL.train(trajs)

learner_trajs = MMCMODEL.unroll_trajectories(transition, 20000, 20)

find_state = lambda x: sw.states[x] if x != MMCMODEL.pad_idx else -1
np_find_state = np.vectorize(find_state)

learner_observations = np_find_state(learner_trajs.numpy())

plot_summary(BC_MMC, trajs, learner_observations, keep_unknown = False)