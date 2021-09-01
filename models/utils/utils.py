import numpy as np
import math
from collections import namedtuple
import torch
import datetime
from tensorboardX import SummaryWriter

from torch.utils.data import Dataset, DataLoader


class model_summary_writer(object):
    def __init__(self, summary_name, env):
        now = datetime.datetime.now()
        self.summary = SummaryWriter(
            logdir='log/' + summary_name + '_{}'.format(now.strftime('%Y%m%d_%H%M%S')))
        self.summary_cnt = 0
        self.env = env


class sequence_data(Dataset):
    def __init__(self, obs, len0, act):
        self.obs = obs
        self.len = len0
        self.act = act

        self.data_size = obs.size(0)

    def __getitem__(self, index):
        return self.obs[index], self.len[index], self.act[index]

    def __len__(self):
        return self.data_size


class sequence_data_vanilla(Dataset):
    def __init__(self, obs, act):
        self.obs = obs
        self.act = act

        self.data_size = obs.size(0)

    def __getitem__(self, index):
        return self.obs[index], self.act[index]

    def __len__(self):
        return self.data_size


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1, 1)


def get_gae(rewards, learner_len, values, gamma, lamda):
    # rewards = learner_rewards[1]
    # learner_len=learner_len[1]
    # values = learner_values[1]
    # gamma = args.gamma
    # lamda = args.lamda

    rewards = torch.Tensor(rewards)
    returns = torch.zeros_like(rewards)
    advants = -1 * torch.ones_like(rewards)

    masks = torch.ones_like(rewards)
    masks[(learner_len-1):] = 0

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, learner_len)):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        running_tderror = rewards[t] + gamma * \
            previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + gamma * \
            lamda * running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants[:learner_len] = (
        advants[:learner_len] - advants[:learner_len].mean()) / advants[:learner_len].std()
    return returns, advants


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def unsqueeze_trajs(learner_observations, learner_actions, learner_len):
    learner_obs = -1 * np.ones((learner_len.sum(), learner_len.max()+1))
    learner_act = np.zeros((learner_len.sum()))
    learner_l = np.zeros((learner_len.sum()))
    cur_idx = 0
    for length0 in range(1, np.max(learner_len)):
        takeidxs = np.where(learner_len >= length0)[0]
        learner_obs[cur_idx:(cur_idx + takeidxs.shape[0]),
                    :length0] = learner_observations[takeidxs, :length0]
        learner_act[cur_idx:(cur_idx + takeidxs.shape[0])
                    ] = learner_actions[takeidxs, (length0-1)]
        learner_l[cur_idx:(cur_idx+takeidxs.shape[0])] = length0
        cur_idx += takeidxs.shape[0]

    idx = learner_l != 0
    learner_obs = learner_obs[idx]
    learner_act = learner_act[idx]
    learner_l = learner_l[idx]

    idx = learner_act != -1
    learner_obs = learner_obs[idx]
    learner_act = learner_act[idx]
    learner_l = learner_l[idx]

    return learner_obs, learner_act, learner_l


def trajs_squeezedtensor(exp_trajs):
    exp_obs = [[x.cur_state for x in episode]+[episode[-1].next_state]
               for episode in exp_trajs]
    exp_act = [[x.action for x in episode] for episode in exp_trajs]
    exp_len = np.array(list(map(len, exp_obs)))

    max_len = max(exp_len)

    expert_observations = np.ones((len(exp_trajs), max_len), np.int32) * -1
    expert_actions = np.ones((len(exp_trajs), max_len-1), np.int32) * -1

    for i in range(len(exp_obs)):
        expert_observations[i, :exp_len[i]] = exp_obs[i]
        expert_actions[i, :(exp_len[i]-1)] = exp_act[i]

    return expert_observations, expert_actions, exp_len


def trajs_to_tensor(exp_trajs):
    np_trajs = []
    for episode in exp_trajs:
        for i in range(1, len(episode)+1):
            np_trajs.append(
                [[x.cur_state for x in episode[:i]], episode[i-1].action])

    expert_len = np.array([len(x[0]) for x in np_trajs])
    maxlen = np.max(expert_len)

    expert_observations = -np.ones(shape=(len(np_trajs), maxlen))
    expert_actions = np.array([x[1] for x in np_trajs])

    expert_len = []
    for i in range(len(np_trajs)):
        temp = np_trajs[i][0]
        expert_observations[i, :len(temp)] = temp
        expert_len.append(len(temp))

    return expert_observations, expert_actions, expert_len


def arr_to_tensor(find_state, device, exp_obs, exp_act, exp_len):
    exp_states = find_state(exp_obs)
    exp_obs = torch.LongTensor(exp_states)
    exp_act = torch.LongTensor(exp_act)
    exp_len = torch.LongTensor(exp_len)
    # exp_len , sorted_idx = exp_len.sort(0,descending = True)
    # exp_obs = exp_obs[sorted_idx]
    # exp_act = exp_act[sorted_idx]
    return exp_obs, exp_act, exp_len


Step = namedtuple('Step', 'cur_state action next_state reward done')


def check_RouteID(episode, routes):
    state_seq = [str(x.cur_state) for x in episode] + \
        [str(episode[-1].next_state)]
    episode_route = "-".join(state_seq)
    if episode_route in routes:
        idx = routes.index(episode_route)
    else:
        idx = -1
    return idx


def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)


def sigmoid(xs):
    """
    sigmoid function
    inputs:
      xs      1d array
    """
    return [1 / (1 + math.exp(-x)) for x in xs]


def identify_routes(trajs):
    num_trajs = len(trajs)
    route_dict = {}
    for i in range(num_trajs):
        episode = trajs[i]
        route = "-".join([str(x.cur_state)
                          for x in episode] + [str(episode[-1].next_state)])
        if route in route_dict.keys():
            route_dict[route] += 1
        else:
            route_dict[route] = 1

    out_list = []
    for key in route_dict.keys():
        route_len = len(key.split("-"))
        out_list.append((key, route_len, route_dict[key]))
    out_list = sorted(out_list, key=lambda x: x[2],  reverse=True)
    return out_list


def expert_compute_state_visitation_freq(sw, trajs):
    feat_exp = np.zeros([sw.n_states])
    for episode in trajs:
        for step in episode:
            feat_exp[sw.pos2idx(step.cur_state)] += 1
        feat_exp[sw.pos2idx(step.next_state)] += 1
    feat_exp = feat_exp/len(trajs)
    return feat_exp


def expert_compute_state_action_visitation_freq(sw, trajs):
    N_STATES = sw.n_states
    N_ACTIONS = sw.max_actions

    mu = np.zeros([N_STATES, N_ACTIONS])

    for episode in trajs:
        for step in episode:
            cur_state = step.cur_state
            s = sw.pos2idx(cur_state)
            action_list = sw.get_action_list(cur_state)
            action = step.action
            a = action_list.index(action)
            mu[s, a] += 1

    mu = mu/len(trajs)
    return mu


def compute_state_visitation_freq(sw, gamma, trajs, policy, deterministic=True):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming

    inputs:
      P_a     NxNxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of list of Steps - collected from expert
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


    returns:
      p       Nx1 vector - state visitation frequencies
    """
    N_STATES = sw.n_states
    # N_ACTIONS = sw.max_actions

    T = len(trajs[0])+1
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    for traj in trajs:
        mu[sw.pos2idx(traj[0].cur_state), 0] += 1
    mu[:, 0] = mu[:, 0]/len(trajs)

    for t in range(T-1):
        for s in range(N_STATES):
            if deterministic:
                mu[s, t+1] = sum([mu[pre_s, t]*sw.is_connected(sw.idx2pos(pre_s), np.argmax(
                    policy[pre_s]), sw.idx2pos(s)) for pre_s in range(N_STATES)])
                # mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, np.argmax(policy[pre_s])] for pre_s in range(N_STATES)])
            else:
                mu_temp = 0
                for pre_s in range(N_STATES):
                    action_list = sw.get_action_list(sw.idx2pos(pre_s))
                    for a1 in range(len(action_list)):
                        mu_temp += mu[pre_s, t]*sw.is_connected(sw.idx2pos(
                            pre_s), action_list[a1], sw.idx2pos(s)) * policy[pre_s, a1]

                mu[s, t+1] = mu_temp
                # mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])

    p = np.sum(mu, 1)
    return p


def compute_state_action_visitation_freq(sw, gamma, trajs, policy, deterministic=True):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming

    inputs:
      P_a     NxNxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of list of Steps - collected from expert
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


    returns:
      p       Nx1 vector - state visitation frequencies
    """

    N_STATES = sw.n_states
    N_ACTIONS = sw.max_actions

    route_list = identify_routes(trajs)
    max_route_length = max([x[1] for x in route_list])
    T = max_route_length

    mu = np.zeros([N_STATES, N_ACTIONS, T])
    start_state = sw.start
    s = sw.pos2idx(start_state)
    action_list = sw.get_action_list(start_state)
    mu[s, :, 0] = policy[s, :]

    for t in range(T-1):
        for s in range(N_STATES):
            action_list = sw.get_action_list(sw.idx2pos(s))
            for a in range(len(action_list)):
                next_state = sw.netconfig[sw.idx2pos(s)][action_list[a]]
                s1 = sw.pos2idx(next_state)
                next_action_list = sw.get_action_list(next_state)
                mu[s1, :len(next_action_list), t+1] += mu[s, a, t] * \
                    policy[s1, :len(next_action_list)]

    p = np.sum(mu, axis=2)
    # p.shape
    return p
