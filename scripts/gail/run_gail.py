import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())

#!/usr/bin/python3
from models.utils.plotutils import *
from models.utils.utils import *
from mdp import shortestpath
from models.gail.algo.gailrnn_pytorch import GAILRNNTrain
from models.gail.network_models.infoq_rnn import infoQ_RNN
from models.gail.network_models.discriminator_rnn import Discriminator as Discriminator_rnn
from models.gail.network_models.discriminator_wgail import Discriminator as Discriminator_wgail
from models.gail.network_models.policy_net_rnn import Policy_net, Value_net, StateSeqEmb
import argparse
# import gym
import time
import os
import numpy as np
import pandas as pd
# import tensorflow as tf
import torch



# from algo.ppo_pytorch import PPOTrain
# import tensorflow as tf


def argparser():
    import sys
    # sys.argv=['']
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e5), type=int)
    parser.add_argument('--n-episode', default=int(20000), type=int)
    parser.add_argument('--num-step1', default=int(1e4), type=int)
    parser.add_argument('--pretrain-step', default=int(0), type=int)
    parser.add_argument('-b', '--batch-size', default=int(8192), type=int)
    parser.add_argument('-nh', '--hidden', default=int(64), type=int)
    parser.add_argument('-ud', '--num-discrim-update',
                        default=int(2), type=int)
    parser.add_argument('-ug', '--num-gen-update', default=int(6), type=int)
    parser.add_argument('-lr', '--learning-rate',
                        default=float(5e-5), type=float)
    parser.add_argument('--c_1', default=float(1), type=float)
    parser.add_argument('--c_2', default=float(0.01), type=float)
    parser.add_argument('--eps', default=float(1e-6), type=float)
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--train-mode', default="value_policy", type=str)
    parser.add_argument('--data', default="data/Single_OD/Binomial.csv", type=str)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('-w', '--wasser', action="store_true", default=False)
    parser.add_argument('-ml', '--max-length', default=int(20), type=int)
    parser.add_argument('--gangnam', default=False, action="store_true")
    return parser.parse_args()


args = argparser()


def main(args):
    if args.gangnam:
        netins = [222, 223, 224, 225, 226, 227, 228,
                  214, 213, 212, 211, 210, 209, 208,
                  190, 189, 188, 187, 186, 185, 184,
                  167, 168, 169, 170, 171, 172, 173, 174, 175, 176]
        netouts = [191, 192, 193, 194, 195, 196, 197,
                   183, 182, 181, 180, 179, 178, 177,
                   221, 220, 219, 218, 217, 216, 215,
                   198, 199, 200, 201, 202, 203, 204, 205, 206, 207]
        env = shortestpath.ShortestPath(
            "data/gangnam_Network.txt", netins, netouts)
    else:
        netins = [252, 273, 298, 302, 372, 443, 441, 409, 430, 392, 321, 245]
        netouts = [253,  276, 301, 299, 376, 447, 442, 400, 420, 393, 322, 246]
        env = shortestpath.ShortestPath("data/Network.txt", netins, netouts)

    exp_trajs = env.import_demonstrations(args.data)
    pad_idx = len(env.states)

    if torch.cuda.is_available() & args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")

    ob_space = env.n_states
    act_space = env.n_actions

    state_dim = ob_space
    action_dim = act_space

    def find_state(x): return env.states.index(x) if x != -1 else pad_idx
    find_state = np.vectorize(find_state)

    origins = np.array(env.origins)
    origins = find_state(origins)
    origins = torch.Tensor(origins).long().to(device)

    policy = Policy_net(state_dim, action_dim,
                        hidden=args.hidden,
                        origins=origins,
                        start_code=env.states.index(env.start),
                        env=env,
                        disttype="categorical")

    value = Value_net(
        state_dim, origins.shape[0], hidden=args.hidden, num_layers=args.num_layers)
    if args.wasser:
        D = Discriminator_wgail(
            state_dim, origins.shape[0], hidden=args.hidden, disttype="categorical", num_layers=args.num_layers)
    else:
        D = Discriminator_rnn(
            state_dim, origins.shape[0], hidden=args.hidden, disttype="categorical", num_layers=args.num_layers)

    if device.type == "cuda":
        policy = policy.cuda()
        value = value.cuda()
        D = D.cuda()

    GAILRNN = GAILRNNTrain(env=env,
                           Policy=policy,
                           Value=value,
                           Discrim=D,
                           pad_idx=pad_idx,
                           args=args)
    GAILRNN.set_device(device)
    if args.wasser:
        GAILRNN.train_discrim_step = GAILRNN.train_wasser_discrim_step
        GAILRNN.discrim_opt = torch.optim.RMSprop(
            GAILRNN.Discrim.parameters(), lr=GAILRNN.lr, eps=GAILRNN.eps)

    if args.pretrain_step > 0:
        GAILRNN.pretrain(exp_trajs, find_state, device, args)

    hard_update(GAILRNN.Value.StateSeqEmb, GAILRNN.Policy.StateSeqEmb)
    hard_update(GAILRNN.Discrim.StateSeqEmb, GAILRNN.Policy.StateSeqEmb)

    for _ in range(args.iteration):

        now = time.time()
        learner_observations, learner_actions, learner_len, learner_rewards =\
            GAILRNN.unroll_trajectory2(
                num_trajs=args.n_episode, max_length=args.max_length)
        # print(time.time() - now)

        mask = learner_rewards != -1
        avg_reward = np.mean(np.sum(learner_rewards * mask, axis=1))
        avg_ind_reward = (learner_rewards * mask).sum() / mask.sum()
        avg_len = learner_len.mean()
        GAILRNN.summary.add_scalar(
            'GenTrajs/reward', avg_reward, GAILRNN.summary_cnt)
        GAILRNN.summary.add_scalar(
            'GenTrajs/step_reward', avg_ind_reward, GAILRNN.summary_cnt)
        GAILRNN.summary.add_scalar(
            'GenTrajs/length', avg_len, GAILRNN.summary_cnt)
        plot_summary(GAILRNN, exp_trajs, learner_observations)

        learner_obs = -1 * np.ones((learner_len.sum(), learner_len.max()))
        learner_act = np.zeros((learner_len.sum()))
        learner_l = np.zeros((learner_len.sum()))
        cnt = 0
        for i0 in range(learner_len.shape[0]):
            for j0 in range(1, learner_len[i0]+1):
                try:
                    learner_obs[cnt, :j0] = learner_observations[i0, :j0]
                    learner_act[cnt] = int(learner_actions[i0][j0-1])
                    learner_l[cnt] = j0
                    cnt += 1
                except:
                    # print("break with index error in Learner Trajectory")
                    break
        idx = learner_l != 0
        learner_obs = learner_obs[idx]
        learner_act = learner_act[idx]
        learner_l = learner_l[idx]
        learner_obs, learner_act, learner_len = arr_to_tensor(
            find_state, device, learner_obs, learner_act, learner_l)

        sample_indices = np.random.randint(
            low=0, high=len(exp_trajs), size=args.n_episode)
        exp_trajs_temp = np.take(a=exp_trajs, indices=sample_indices, axis=0)
        exp_obs, exp_act, exp_len = trajs_to_tensor(exp_trajs_temp)
        exp_obs, exp_act, exp_len = arr_to_tensor(
            find_state, device, exp_obs, exp_act, exp_len)

        GAILRNN.train(exp_obs=exp_obs,
                      exp_act=exp_act,
                      exp_len=exp_len,
                      learner_obs=learner_obs,
                      learner_act=learner_act,
                      learner_len=learner_len)

        elapsed = time.time()-now
        GAILRNN.summary.add_scalar(
            'GenTrajs/elapsed_time', elapsed, GAILRNN.summary_cnt)
        print("Total Reward = {:.2f} / Ind. Reward = {:.5f} / Length = {:.2f} / ElapsedTime = {:.2f}".format(
            avg_reward, avg_ind_reward, avg_len, elapsed))


if __name__ == '__main__':
    args = argparser()
    print(args)
    main(args)
