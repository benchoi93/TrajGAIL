



import matplotlib.pyplot as plt
import numpy as np
import math
import io
import pandas as pd
from models.utils.utils import identify_routes,check_RouteID
import time

def plot_summary(GAIL, exp_trajs , learner_observations):
    expert   = [episode[0].next_state for episode in exp_trajs]
    # gail     = [episode[0].next_state for episode in learner_trajs]
    gail     = list(learner_observations[:,1])
    route_idxs= GAIL.env.origins
    route_dist = [(i ,expert.count(i) , gail.count(i)) for i in route_idxs]
    np_route_dist = np.array(route_dist,np.float64)
    np_route_dist[:,1:] = np_route_dist[:,1:] / np.sum(np_route_dist[:,1:],0)
    pd_route_dist = pd.DataFrame(np_route_dist)
    pd_route_dist.columns = ["routeid",'expert','gail']
    fig,ax = plot_barchart(pd_route_dist, route_idxs, "compare.png", keep_unknown=False)
    GAIL.summary.add_figure("Origin_Distribution", fig , GAIL.summary_cnt)

    expert   = [episode[-1].cur_state for episode in exp_trajs]
    gail     = list(learner_observations[np.arange(learner_observations.shape[0]) , np.sum(learner_observations != -1 , axis =1)-1])
    route_idxs= GAIL.env.destinations
    route_dist = [(i ,expert.count(i) , gail.count(i)) for i in route_idxs]
    np_route_dist = np.array(route_dist,np.float64)

    if 0 in np.sum(np_route_dist[:,1:],0):
        print(1)

    np_route_dist[:,1:] = np_route_dist[:,1:] / np.sum(np_route_dist[:,1:],0)
    pd_route_dist = pd.DataFrame(np_route_dist)
    pd_route_dist.columns = ["routeid",'expert','gail']
    fig,ax = plot_barchart(pd_route_dist, route_idxs, "compare.png", keep_unknown=False)
    GAIL.summary.add_figure("Destination_Distribution", fig , GAIL.summary_cnt)
    
    expert_routes = identify_routes(exp_trajs)
    routes = [x[0] for x in expert_routes]
    expert = [check_RouteID(episode, routes) for episode in exp_trajs]
    # gail   = [check_RouteID(episode, routes) for episode in learner_trajs]
    gail = []
    for x in learner_observations:
        x = list(x)    
        while -1 in x:
            x.remove(-1)
        x = list(map(str, x))
        episode_route = "-".join(x)      
        if episode_route in routes:
            idx = routes.index(episode_route)
        else:
            idx = -1
        gail.append(idx)

    route_idxs= list(range(len(routes)))+[-1]
    route_dist = [(i ,expert.count(i) , gail.count(i)) for i in route_idxs]
    np_route_dist = np.array(route_dist,np.float64)
    np_route_dist[:,1:] = np_route_dist[:,1:] / np.sum(np_route_dist[:,1:],0)
    pd_route_dist = pd.DataFrame(np_route_dist)
    pd_route_dist.columns = ["routeid",'expert','gail']
    if len(routes) > 15:
        fig,ax = plot_linechart(pd_route_dist, route_idxs, "compare.png", keep_unknown=True)
    else:
        fig,ax = plot_barchart(pd_route_dist, route_idxs, "compare.png", keep_unknown=True)
    GAIL.summary.add_figure("Route_Distribution", fig , GAIL.summary_cnt)

    # learner_svf = sum([[x.cur_state for x in episode] for episode in learner_trajs] , []) + [episode[-1].next_state for episode in learner_trajs]

    learner_svf = []
    for x in learner_observations:
        x = list(x)    
        while -1 in x:
            x.remove(-1)
        x = list(map(int, x))
        learner_svf.append(x)
    learner_svf = sum(learner_svf , [])
    expert_svf   = sum([[x.cur_state for x in episode] for episode in exp_trajs] , []) + [episode[-1].next_state for episode in exp_trajs]
    svf = [(i ,expert_svf.count(i) , learner_svf.count(i)) for i in GAIL.env.states]
    svf = np.array(svf,np.float)
    svf[:,1:] = svf[:,1:].astype(np.float) / np.array([len(exp_trajs) , learner_observations.shape[0]] , np.float) * 100
    pd_svf = pd.DataFrame(svf)
    pd_svf.columns = ["StateID","expert","GAIL"]

    fig,ax = plt.subplots()
    ax.scatter(pd_svf["GAIL"] , pd_svf["expert"] , color ="b")
    ax.plot([0,100] ,[0,100] , color = "r")
    ax.set_xlabel("GAIL")
    ax.set_ylabel("expert")
    GAIL.summary.add_figure("SVF", fig , GAIL.summary_cnt)


def plot_summary_maxent(GAIL, exp_trajs , learner_trajs):

    expert   = [(episode[0].next_state , episode[-1].cur_state) for episode in exp_trajs]
    gail   = [(episode[0].next_state , episode[-1].cur_state) for episode in learner_trajs]
    # gail_o   = list(learner_observations[:,1])
    # gail_d   = list(learner_observations[np.arange(learner_observations.shape[0]) , np.sum(learner_observations != -1 , axis =1)-2])
    # gail     = [x for x in zip(gail_o,gail_d)]

    # route_idxs= list(set(expert))    

    np_expert = np.array(expert)
    np_gail   = np.array(gail)

    np_expert_unq, np_expert_cnt = np.unique(np_expert,return_counts=True , axis =0)
    np_gail_unq, np_gail_cnt = np.unique(np_gail,return_counts=True , axis =0)

    route_idxs = [list(x) for x in list(np_expert_unq)]
    
    route_dist =  [(i, np_expert_cnt[i] , np_gail_cnt[np.where(np.all(np_gail_unq == np_expert_unq[i] , axis=1))[0][0]] )  for i in range(np_expert_unq.shape[0]) if np.where(np.all(np_gail_unq == np_expert_unq[i] , axis=1))[0].shape[0] > 0]
    route_dist += [(np_expert_unq.shape[0] , 0, np.sum(np_gail_cnt)-sum([x[2] for x in route_dist]))]

    np_route_dist = np.array(route_dist,np.float64)
    np_route_dist[:,1:] = np_route_dist[:,1:] / np.sum(np_route_dist[:,1:],0)
    pd_route_dist = pd.DataFrame(np_route_dist)
    pd_route_dist.columns = ["routeid",'expert','gail']

    if pd_route_dist.shape[0] > 15:
        pd_route_dist = pd_route_dist.sort_values(by=["expert"] , ascending = False)
        fig,ax = plot_linechart(pd_route_dist, list(pd_route_dist['routeid']), "compare.png")
    else:
        fig,ax = plot_barchart(pd_route_dist, list(pd_route_dist['routeid']), "compare.png")
    GAIL.summary.add_figure("OD_distribution", fig , GAIL.summary_cnt)


    expert   = [episode[0].next_state for episode in exp_trajs]
    gail     = [episode[0].next_state for episode in learner_trajs]
    # gail     = list(learner_observations[:,1])
    route_idxs= GAIL.env.origins
    route_dist = [(i ,expert.count(i) , gail.count(i)) for i in route_idxs]
    np_route_dist = np.array(route_dist,np.float64)
    np_route_dist[:,1:] = np_route_dist[:,1:] / np.sum(np_route_dist[:,1:],0)
    pd_route_dist = pd.DataFrame(np_route_dist)
    pd_route_dist.columns = ["routeid",'expert','gail']
    fig,ax = plot_barchart(pd_route_dist, route_idxs, "compare.png", keep_unknown=False)
    GAIL.summary.add_figure("Origin_Distribution", fig , GAIL.summary_cnt)

    expert   = [episode[-1].cur_state for episode in exp_trajs]
    gail   = [episode[-1].cur_state for episode in learner_trajs]
    # gail     = list(learner_observations[np.arange(learner_observations.shape[0]) , np.sum(learner_observations != -1 , axis =1)-1])
    route_idxs= GAIL.env.destinations
    route_dist = [(i ,expert.count(i) , gail.count(i)) for i in route_idxs]
    np_route_dist = np.array(route_dist,np.float64)
    np_route_dist[:,1:] = np_route_dist[:,1:] / np.sum(np_route_dist[:,1:],0)
    pd_route_dist = pd.DataFrame(np_route_dist)
    pd_route_dist.columns = ["routeid",'expert','gail']
    fig,ax = plot_barchart(pd_route_dist, route_idxs, "compare.png", keep_unknown=False)
    GAIL.summary.add_figure("Destination_Distribution", fig , GAIL.summary_cnt)
    
    expert_routes = identify_routes(exp_trajs)
    routes = [x[0] for x in expert_routes]
    expert = [check_RouteID(episode, routes) for episode in exp_trajs]
    gail   = [check_RouteID(episode, routes) for episode in learner_trajs]
    # gail = []
    # for x in learner_observations:
    #     x = list(x)    
    #     while -1 in x:
    #         x.remove(-1)
    #     x = list(map(str, x))
    #     episode_route = "-".join(x)      
    #     if episode_route in routes:
    #         idx = routes.index(episode_route)
    #     else:
    #         idx = -1
    #     gail.append(idx)

    route_idxs= list(range(len(routes)))+[-1]
    route_dist = [(i ,expert.count(i) , gail.count(i)) for i in route_idxs]
    np_route_dist = np.array(route_dist,np.float64)
    np_route_dist[:,1:] = np_route_dist[:,1:] / np.sum(np_route_dist[:,1:],0)
    pd_route_dist = pd.DataFrame(np_route_dist)
    pd_route_dist.columns = ["routeid",'expert','gail']
    if len(routes) > 15:
        fig,ax = plot_linechart(pd_route_dist, route_idxs, "compare.png", keep_unknown=True)
    else:
        fig,ax = plot_barchart(pd_route_dist, route_idxs, "compare.png", keep_unknown=True)
    GAIL.summary.add_figure("Route_Distribution", fig , GAIL.summary_cnt)

    expert_svf   = sum([[x.cur_state for x in episode] for episode in exp_trajs] , []) + [episode[-1].next_state for episode in exp_trajs]
    learner_svf   = sum([[x.cur_state for x in episode] for episode in learner_trajs] , []) + [episode[-1].next_state for episode in learner_trajs]

    svf = [(i ,expert_svf.count(i) , learner_svf.count(i)) for i in GAIL.env.states]
    svf = np.array(svf,np.float)
    svf[:,1:] = svf[:,1:].astype(np.float) / np.array([len(exp_trajs) , len(learner_trajs) ] , np.float) * 100
    pd_svf = pd.DataFrame(svf)
    pd_svf.columns = ["StateID","expert","GAIL"]

    fig,ax = plt.subplots()
    ax.scatter(pd_svf["GAIL"] , pd_svf["expert"] , color ="b")
    ax.plot([0,100] ,[0,100] , color = "r")
    ax.set_xlabel("GAIL")
    ax.set_ylabel("expert")
    GAIL.summary.add_figure("SVF", fig , GAIL.summary_cnt)


def plot_linechart(df,route_idxs, outpath, keep_unknown = False):
    labels = ["R"+str(i+1) for i in range(len(route_idxs)-1)] + ["unknown"]
    if not keep_unknown:
        df = df.loc[df["routeid"]!=-1]
        route_idxs = [x for x in route_idxs if x != -1]
        labels = [x for x in labels if x != "unknown"]

    num_models = df.shape[1]-1
    fig,ax = plt.subplots()
    fig.set_size_inches(10,10)
    X = np.arange(len(route_idxs))

    cmap=plt.cm.rainbow(np.linspace(0, 1, num_models)) 
    bar_width = 1/(num_models+1)
    num_front = math.ceil((num_models-1)/2)

    for i in range(0,num_models):
        ax.plot(X , df[df.columns[i+1]] , color = cmap[i])

    ax.legend(labels=df.columns[1:])
    locs , labels = plt.xticks(ticks=X , labels = labels)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Route")
    # plt.savefig(outpath , dpi=300 )
    return fig,ax

def plot_barchart(df,route_idxs, outpath, keep_unknown = False):
    labels = ["R"+str(i+1) for i in range(len(route_idxs)-1)] + ["unknown"]
    if not keep_unknown:
        df = df.loc[df["routeid"]!=-1]
        route_idxs = [x for x in route_idxs if x != -1]
        labels = [x for x in labels if x != "unknown"]

    num_models = df.shape[1]-1
    fig,ax = plt.subplots()
    fig.set_size_inches(10,10)
    X = np.arange(len(route_idxs))

    cmap=plt.cm.rainbow(np.linspace(0, 1, num_models)) 
    bar_width = 1/(num_models+1)
    num_front = math.ceil((num_models-1)/2)

    for i in range(0,num_models):
        ax.bar(X-num_front*bar_width+i*bar_width , df[df.columns[i+1]] , color = cmap[i],width = bar_width )
    ax.legend(labels=df.columns[1:])
    locs , labels = plt.xticks(ticks=X , labels = labels)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Route")
    # plt.savefig(outpath , dpi=300 )
    return fig,ax

def plot_seperate_barchart(df,route_idxs,outpath,keep_unknown=False):
    labels = ["R"+str(i+1) for i in range(len(route_idxs)-1)] + ["unknown"]
    if not keep_unknown:
        df = df.loc[df["routeid"]!=-1]
        route_idxs = [x for x in route_idxs if x != -1]
        labels = [x for x in labels if x != "unknown"]


    num_models = df.shape[1]-1
    fig,ax = plt.subplots()
    fig.set_size_inches(30,10)
    X = np.arange(len(route_idxs))
    cmap=plt.cm.rainbow(np.linspace(0, 1, num_models)) 

    modelnames = df.columns[1:]
    for i in range(len(modelnames)):
        model = modelnames[i]
        fig,ax = plt.subplots()
        X = np.arange(len(route_idxs))
        ax.bar(X , df[model] , color = cmap[i], width = 0.2)
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Route")
        ax.set_title(model)
        plt.savefig(outpath+"_"+model+".png" , dpi=300 )
