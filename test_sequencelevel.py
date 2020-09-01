import os
import sys

import numpy as np
import argparse
from mdp import shortestpath

from models.utils.utils import *
from models.utils.plotutils import *
from models.irl.algo import value_iteration
from models.behavior_clone.mmc_predictor import MMC_predictor
from models.behavior_clone.rnn_predictor import *

# sys.argv=[""]
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gangnam', default = False, action = "store_true" )
    parser.add_argument("--max-length",default = 15,type=int)
    parser.add_argument("--num-trajs",default=1000,type=int)
    return parser.parse_args()
args = argparser()


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
    list_all_data = [os.path.join("data","gangnam_expert.csv")]
else:
    origins = [252, 273, 298, 302, 372, 443, 441, 409, 430, 392, 321, 245 ]
    destinations = [ 253,  276, 301, 299, 376, 447, 442, 400, 420, 393, 322, 246]
    sw = shortestpath.ShortestPath("data/Network.txt",origins, destinations)
    dataset_list = ["Single_OD","Multi_OD"]
    list_all_data = []
    for dataset in dataset_list:
        for data0 in os.listdir(os.path.join("data",dataset)):
            list_all_data.append(os.path.join("data",dataset,data0))

N_STATES = sw.n_states

# list_all_data = list_all_data[13:]

for data0 in list_all_data:
    
    print("current data :: {}".format(data0))
    # data0=list_all_data[0]
    trajs = sw.import_demonstrations(data0)

    datainfo = data0.split(os.sep)
    if args.gangnam:
        datainfo[-2]="gangnam"
        datainfo[-1]="dtg"
        mmc_model = np.load( os.path.join("Result",datainfo[-2],'transition_{}_{}.npy'.format(datainfo[-2],datainfo[-1].split(".")[0])))
        svf_model = np.load(os.path.join("Result",datainfo[-2],"rewards_maxent.npy"))
        savf_model = np.load(os.path.join("Result",datainfo[-2],"rewards_maxent_state_action.npy"))
    else:
        mmc_model = np.load( os.path.join("Result",datainfo[-2],'transition_{}_{}.npy'.format(datainfo[-2],datainfo[-1].split(".")[0])))
        svf_model = np.load(os.path.join("Result",datainfo[-2],datainfo[-1],"rewards_maxent.npy"))
        savf_model = np.load(os.path.join("Result",datainfo[-2],datainfo[-1],"rewards_maxent_state_action.npy"))

    _,svf_model_policy = value_iteration.value_iteration(sw,svf_model,0.5)
    _,savf_model_policy = value_iteration.action_value_iteration(sw,savf_model,0.5)

    MMCMODEL = MMC_predictor(sw, args.max_length)
    generated_mmc = MMCMODEL.unroll_trajectories(torch.Tensor(mmc_model),args.num_trajs,args.max_length)
    find_state = lambda x: sw.states[x] if x != MMCMODEL.pad_idx else -1
    np_find_state = np.vectorize(find_state)
    generated_mmc = np_find_state(generated_mmc.numpy())

    generated_svf = sw.generate_demonstrations(svf_model_policy , n_trajs = args.num_trajs)
    generated_savf = sw.generate_demonstrations(savf_model_policy , n_trajs = args.num_trajs)

    RNNMODEL = RNN_predictor(sw.states, 256,pad_idx = -1)
    RNNMODEL.load_state_dict(torch.load(os.path.join("Result",datainfo[-2],'RNN_{}_{}.pth'.format(datainfo[-2],datainfo[-1].split(".")[0]))))

    generated_rnn = RNNMODEL.unroll_trajectories(sw.start, sw.terminal,args.num_trajs,args.max_length)
    find_state = lambda x: RNNMODEL.states[x]
    np_find_state= np.vectorize(find_state)
    generated_rnn = np_find_state(generated_rnn.numpy())


    print("all sequences generated, scoring start")

    import nltk

    exp_trajs = [[i.cur_state for i in x]+[x[-1].next_state] for x in trajs]


    generated_mmc = [[x for x in list(x) if x != -1] for x in generated_mmc]
    generated_svf = [[i.cur_state for i in x]+[x[-1].next_state] for x in generated_svf]
    generated_savf = [[i.cur_state for i in x]+[x[-1].next_state] for x in generated_savf]
    generated_rnn = [[x for x in list(x) if x != -1] for x in generated_rnn]

    def test_bleu(i):
        return (nltk.translate.bleu(exp_trajs,generated_mmc[i]),
                nltk.translate.bleu(exp_trajs,generated_svf[i]),
                nltk.translate.bleu(exp_trajs,generated_savf[i]),
                nltk.translate.bleu(exp_trajs,generated_rnn[i])
                ) 

    def test_meteor(i):
        temp_exp_trajs = [" ".join(map(str,x)) for x in exp_trajs]
        seqtostring= lambda x : " ".join(map(str,x))

        return (nltk.translate.meteor_score.meteor_score(temp_exp_trajs,seqtostring(generated_mmc[i])),
                nltk.translate.meteor_score.meteor_score(temp_exp_trajs,seqtostring(generated_svf[i])),
                nltk.translate.meteor_score.meteor_score(temp_exp_trajs,seqtostring(generated_savf[i])),
                nltk.translate.meteor_score.meteor_score(temp_exp_trajs,seqtostring(generated_rnn[i]))
                ) 


    bleu_result = np.zeros(shape=(args.num_trajs , 4))
    meteor_result = np.zeros(shape=(args.num_trajs , 4))
    # for i in range(100) :
    for i in range(args.num_trajs) :
        bleu_result[i,:] = test_bleu(i)
        meteor_result[i,:] = test_meteor(i)
        print(i)

    np.savetxt(os.path.join("Result","BLEUresult_{}_{}.csv".format(datainfo[-2],datainfo[-1].split(".")[0])), bleu_result, delimiter=",")
    np.savetxt(os.path.join("Result","METEORresult_{}_{}.csv".format(datainfo[-2],datainfo[-1].split(".")[0])), meteor_result, delimiter=",")

    print("BLEU score saved at : " + os.path.join("Result","BLEUresult_{}_{}.csv".format(datainfo[-2],datainfo[-1].split(".")[0])))
    print("METEOR score saved at :" + os.path.join("Result","METEORresult_{}_{}.csv".format(datainfo[-2],datainfo[-1].split(".")[0])))