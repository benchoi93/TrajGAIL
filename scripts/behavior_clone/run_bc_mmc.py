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

from models.behavior_clone.rnn_predictor import *

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr','--learning-rate', default=0.1, type=float)
    parser.add_argument('-g','--gamma', default=0.5, type=float)
    parser.add_argument('-n','--n-iters',default=int(1000),type =int)
    parser.add_argument('-hd','--hidden',default=int(256),type =int)
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


BC_MMC = model_summary_writer("test_BC_RNN_"+dataname,sw)

max_length = max(list(map(len , trajs)))
PAD_IDX = -1

trajs_list = [[x.cur_state for x in episode ] + [episode[-1].next_state] + [-1]*(max_length - len(episode)) for episode in trajs] 
trajs_np = np.array(trajs_list)

RNNMODEL = RNN_predictor(sw.states, args.hidden,pad_idx = PAD_IDX)

idx_seq = RNNMODEL.states_to_idx(trajs_np)

# self = RNNMODEL
seq = torch.LongTensor(idx_seq)

x_train = seq[:, :(seq.size(1)-1)]
y_train = seq[:, 1:]

criterion = torch.nn.CrossEntropyLoss(ignore_index=RNNMODEL.pad_idx)
optimizer = torch.optim.Adam(RNNMODEL.parameters(),lr = args.learning_rate)

for _ in range(args.n_iters):
    y_est = RNNMODEL(x_train)
    loss = criterion(y_est.view((y_train.size(0)*y_train.size(1)) , y_est.size(2)), y_train.contiguous().view((y_train.size(0)*y_train.size(1),))  )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    learner_observations = RNNMODEL.unroll_trajectories(sw.start,sw.terminal,2000,20)
    find_state = lambda x: RNNMODEL.states[x]
    np_find_state = np.vectorize(find_state)
    learner_observations = np_find_state(learner_observations.numpy())

    plot_summary(BC_RNN, trajs, learner_observations)
    BC_RNN.summary.add_scalar("loss",loss.item())
    BC_RNN.summary_cnt +=1