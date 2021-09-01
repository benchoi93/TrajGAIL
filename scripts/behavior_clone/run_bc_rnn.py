import os 
import sys

# os.chdir('D:\TrajGen_GAIL')
# sys.argv=['']

sys.path.append(os.getcwd())
from pathlib import Path

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
    parser.add_argument('-lr','--learning-rate', default=0.01, type=float)
    parser.add_argument('-g','--gamma', default=0.5, type=float)
    parser.add_argument('-n','--n-iters',default=int(1000),type =int)
    parser.add_argument('-hd','--hidden',default=int(256),type =int)
    parser.add_argument('-pf','--print-freq',default=int(20),type =int)
    parser.add_argument('-b','--batch-size',default=int(256),type =int)
    # parser.add_argument('-d' , '--data', default = "data/Multi_OD/One_way_Binomial.csv")
    parser.add_argument('-d' , '--data', default = "data/Single_OD/Binomial.csv")
    parser.add_argument("--cuda", default = True, type =bool)
    parser.add_argument('--gangnam', default = False, action = "store_true" )
    return parser.parse_args()
args = argparser()

if torch.cuda.is_available() & args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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


# BC_RNN = model_summary_writer("test_BC_RNN_{}_{}".format(demand_pattern,dataname),sw)
# BC_RNN = model_summary_writer("{}/test_BC_RNN_{}_{}".format(dataname,dataname, demand_type),sw)
BC_RNN = model_summary_writer("{}/{}/test_BC_RNN_{}_{}".format(dataname, demand_type,dataname, demand_type),sw)

max_length = max(list(map(len , trajs)))
PAD_IDX = -1

num_train = int(0.7 * len(trajs))
num_test = int(0.3 * len(trajs))
data_idxs = np.random.permutation(len(trajs))

train_idxs = data_idxs[:num_train]
test_idxs = data_idxs[num_train:(num_train+num_test)]

train_trajs = [trajs[i] for i in train_idxs]
test_trajs = [trajs[i] for i in test_idxs]


trajs_list = [[x.cur_state for x in episode ] + [episode[-1].next_state] + [-1]*(max_length - len(episode)) for episode in trajs] 
trajs_np = np.array(trajs_list)

RNNMODEL = RNN_predictor(sw.states, args.hidden,pad_idx = PAD_IDX)

idx_seq = RNNMODEL.states_to_idx(trajs_np)

# self = RNNMODEL
seq = torch.LongTensor(idx_seq)

x_train = seq[train_idxs, :(seq.size(1)-1)]
y_train = seq[train_idxs, 1:]

x_test = seq[test_idxs, :(seq.size(1)-1)]
y_test = seq[test_idxs, 1:]

criterion = torch.nn.NLLLoss(ignore_index=RNNMODEL.pad_idx)
optimizer = torch.optim.Adam(RNNMODEL.parameters(),lr = args.learning_rate)

if device.type == "cuda":
    RNNMODEL = RNNMODEL.cuda()
    RNNMODEL.device= torch.device("cuda")

print("training start")

for _ in range(args.n_iters):

    idxs=  np.random.permutation(x_train.shape[0])

    loss_list = []
    for i in range(int(len(idxs) / args.batch_size)):
        #TRAINING
        batch_idxs = idxs[(i)*args.batch_size : (i+1)*args.batch_size]
        
        sampled_x_train = x_train[batch_idxs]
        sampled_y_train = y_train[batch_idxs]

        y_est = RNNMODEL(sampled_x_train.to(device))
        
        loss = criterion(y_est.view((sampled_y_train.size(0)*sampled_y_train.size(1)) , y_est.size(2)), \
                        sampled_y_train.contiguous().to(device).view((sampled_y_train.size(0)*sampled_y_train.size(1),))  )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    acc_list = []
    acc2_list = []
    idxs=  np.random.permutation(x_test.shape[0])
    for i in range(int(len(idxs) / args.batch_size)):
        #TESTING
        batch_idxs = idxs[(i)*args.batch_size : (i+1)*args.batch_size]
        
        sampled_x_test = x_test[batch_idxs]
        sampled_y_test = y_test[batch_idxs]

        y_est = RNNMODEL(sampled_x_test.to(device))

        pred = torch.argmax(y_est,dim=2) == sampled_x_test.to(device)
        acc = pred.float().mean()
        
        prob = y_est.exp()
        prob = prob.view((prob.size(0) * prob.size(1) , prob.size(2)))

        prob_idxs = sampled_y_test
        prob_idxs = sampled_y_test.view(prob.size(0))

        idxs0 = torch.arange(0,prob.shape[0]).to(device)
        acc2 = torch.mean(prob[idxs0][idxs0, prob_idxs[idxs0]])
        
        acc2_list.append(acc2.item())
        acc_list.append(acc.item())

    learner_observations = RNNMODEL.unroll_trajectories(sw.start,sw.terminal,2000,20)
    find_state = lambda x: RNNMODEL.states[x]
    np_find_state = np.vectorize(find_state)
    if learner_observations.is_cuda:
        learner_observations = learner_observations.cpu()
    learner_observations = np_find_state(learner_observations.numpy())

    print("Loss :: {}".format(np.mean(loss_list)))
    plot_summary(BC_RNN, test_trajs, learner_observations)
    BC_RNN.summary.add_scalar("result/loss",np.mean(loss_list),BC_RNN.summary_cnt)
    BC_RNN.summary.add_scalar("result/acc",np.mean(acc_list),BC_RNN.summary_cnt)
    BC_RNN.summary.add_scalar("result/acc2",np.mean(acc2_list),BC_RNN.summary_cnt)
    BC_RNN.summary_cnt +=1

torch.save(RNNMODEL.state_dict() ,
            "{}/{}/RNN_{}_{}.pth".format(\
                dirName,dataname,dataname, demand_type.split('.')[0]
                                                ) 
            )

print("The model is saved at :: {}/{}/RNN_{}_{}.pth ".format(\
                dirName,dataname,dataname, demand_type.split('.')[0]
                                                ) )


from matplotlib import pyplot
pyplot.cla()

expert_lengths = list(map(lambda x : len(x) , trajs))
learner_lengths = np.apply_along_axis(lambda x:np.count_nonzero(x+1) , 1, learner_observations)

maxlen  = max(max(expert_lengths),max(learner_lengths))
minlen  = min(min(expert_lengths),min(learner_lengths))

from matplotlib import pyplot
pyplot.hist(expert_lengths,range(minlen,maxlen+1), alpha=0.5, label='expert',density=True)
pyplot.hist(learner_lengths-1, range(minlen,maxlen+1), alpha=0.5, label='learner',density=True)
pyplot.yscale('log')
pyplot.xlabel("Length")
pyplot.ylabel("Frequency")
pyplot.legend(loc='upper right')
pyplot.xticks(range(minlen,maxlen+1))
Path(f"{dirName}/plot").mkdir(parents=True, exist_ok=True)
pyplot.savefig("{}/plot/{}_{}_RNN.png".format(dirName,dataname,demand_type.split('.')[0]))

