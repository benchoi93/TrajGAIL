import os 
import sys

# os.chdir("D:\\TrajGen_GAIL")
# sys.argv=['','--data', 'data\\Multi_OD\\One_way_Binomial.csv']

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
    parser.add_argument('--gangnam', default = False, action = "store_true" )
    return parser.parse_args()
args = argparser()

# def main(data0, outpath,lr=0.1 , n_iters=1000 , print_freq=20, gamma=0.5 ):
LEARNING_RATE = args.learning_rate
N_ITERS = args.n_iters
PRINT_FREQ = args.print_freq
GAMMA = args.gamma
data0 = args.data

if args.gangnam:
    dataname = "gangnam"
    demand_type = "dtg"
else:
    data_split = []
    splited = data0
    while True:
        splited = os.path.split(splited)
        if len(splited[1]) == 0 :
            break
        else:
            data_split.append(splited[1])
            splited = splited[0]

    dataname = data_split[1]
    demand_type = data_split[0]

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

# MaxEnt_svf = model_summary_writer("test_MaxEnt_svf_{}_{}".format(demand_pattern,dataname),sw)
# MaxEnt_savf = model_summary_writer("test_MaxEnt_savf_{}_{}".format(demand_pattern,dataname),sw)

num_train = int(0.7 * len(trajs))
num_test = int(0.3 * len(trajs))
data_idxs = np.random.permutation(len(trajs))

train_idxs = data_idxs[:num_train]
test_idxs = data_idxs[num_train:(num_train+num_test)]

train_trajs = [trajs[i] for i in train_idxs]
test_trajs = [trajs[i] for i in test_idxs]

MaxEnt_svf = model_summary_writer("{}/{}/test_MaxEnt_svf_{}_{}".format(dataname,demand_type,dataname, demand_type),sw)
MaxEnt_savf = model_summary_writer("{}/{}/test_MaxEnt_savf_{}_{}".format(dataname,demand_type,dataname, demand_type),sw)

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

if not os.path.exists(os.path.join(dirName,dataname,demand_type)):
    os.mkdir(os.path.join(dirName,dataname,demand_type))
    print("Directory " + os.path.join(dirName,dataname,demand_type) +  " Created ")
else:    
    print("Directory " + os.path.join(dirName,dataname,demand_type) +  " already exists")      


route_list = identify_routes(trajs)

####################
### Train MaxEnt ###
if not os.path.exists(os.path.join(dirName,dataname,demand_type,"rewards_maxent.npy")):
    rewards_maxent = maxent_irl(sw,feat_map, GAMMA, train_trajs, LEARNING_RATE*2, N_ITERS*2 , print_freq=PRINT_FREQ)
    np.save( os.path.join(dirName,dataname,demand_type,"rewards_maxent.npy"),rewards_maxent)

########################################
### Train MaxEnt State action###

if not os.path.exists(os.path.join(dirName,dataname,demand_type,"rewards_maxent_state_action.npy")):
    rewards_maxent_state_action = maxent_irl_stateaction(sw,feat_map, GAMMA, train_trajs, LEARNING_RATE*2, N_ITERS*2 , print_freq=PRINT_FREQ)
    np.save( os.path.join(dirName,dataname,demand_type,"rewards_maxent_state_action.npy")  ,rewards_maxent_state_action)

############################################
# Test
rewards_maxent = np.load(os.path.join(dirName,dataname,demand_type,"rewards_maxent.npy"))
rewards_maxent_state_action = np.load(os.path.join(dirName,dataname,demand_type,"rewards_maxent_state_action.npy"))

_, policy = value_iteration.value_iteration(sw, rewards_maxent, GAMMA, error=0.01)
generated_maxent = sw.generate_demonstrations(policy , n_trajs = 10000)

_, policy = value_iteration.action_value_iteration(sw, rewards_maxent_state_action, GAMMA, error=0.01)
generated_maxent_stateaction = sw.generate_demonstrations(policy , n_trajs = 10000)

_, svf_policy = value_iteration.value_iteration(sw, rewards_maxent,GAMMA )
_, savf_policy = value_iteration.action_value_iteration(sw, rewards_maxent_state_action,GAMMA )

svf_acc_list = []
svf_acc2_list = []

savf_acc_list = []
savf_acc2_list = []

for episode in test_trajs:
    for step in episode:
        
        cur_idx = sw.pos2idx(step.cur_state)
        action_list = np.array(sw.get_action_list(step.cur_state))
        action_prob = svf_policy[cur_idx, action_list]

        acc = np.argmax(action_prob) == step.action
        acc2 = action_prob[np.where(action_list == step.action)[0]]

        svf_acc_list.append( acc )
        svf_acc2_list.append( acc2[0] )

        action_prob = savf_policy[cur_idx, action_list]

        acc = np.argmax(action_prob) == step.action
        acc2 = action_prob[np.where(action_list == step.action)[0]]

        savf_acc_list.append( acc )
        savf_acc2_list.append( acc2[0] )

svf_acc_list = np.array(svf_acc_list , np.float64)
svf_acc2_list = np.array(svf_acc2_list , np.float64)

savf_acc_list  = np.array(savf_acc_list , np.float64)
savf_acc2_list = np.array(savf_acc2_list , np.float64)

MaxEnt_svf.summary.add_scalar("result/acc",np.mean(svf_acc_list) , MaxEnt_svf.summary_cnt)
MaxEnt_svf.summary.add_scalar("result/acc2",np.mean(svf_acc2_list) , MaxEnt_svf.summary_cnt)
plot_summary_maxent(MaxEnt_svf, trajs , generated_maxent)

MaxEnt_savf.summary.add_scalar("result/acc",np.mean(savf_acc_list)   , MaxEnt_savf.summary_cnt)
MaxEnt_savf.summary.add_scalar("result/acc2",np.mean(savf_acc2_list) , MaxEnt_savf.summary_cnt)
plot_summary_maxent(MaxEnt_savf, trajs , generated_maxent_stateaction)

MaxEnt_svf.summary_cnt+=N_ITERS
MaxEnt_savf.summary_cnt+=N_ITERS

MaxEnt_svf.summary.add_scalar("result/acc",np.mean(svf_acc_list  ) , MaxEnt_svf.summary_cnt)
MaxEnt_svf.summary.add_scalar("result/acc2",np.mean(svf_acc2_list) , MaxEnt_svf.summary_cnt)
plot_summary_maxent(MaxEnt_svf, trajs , generated_maxent)

MaxEnt_savf.summary.add_scalar("result/acc",np.mean(savf_acc_list  ) , MaxEnt_savf.summary_cnt)
MaxEnt_savf.summary.add_scalar("result/acc2",np.mean(savf_acc2_list) , MaxEnt_savf.summary_cnt)
plot_summary_maxent(MaxEnt_savf, trajs , generated_maxent_stateaction)





