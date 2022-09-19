#coding: utf-8

from params import *
import algorithms
import problems
import betas
from data import create_save_instances
from utils import boxplot

## Reproducibility
import numpy as np
from random import seed as randomseed
from joblib import Parallel, delayed
randomseed(seed_nb)
np.random.seed(seed_nb)

import subprocess as sb
import pandas as pd
from glob import glob
import json
import argparse
import pickle

parser = argparse.ArgumentParser(description='Near-Optimal Federated Learning')
parser.add_argument('--params_path', type=str, help="Path of JSON parameter file params.json in results/", default="")
parser.add_argument('--bandit_type', type=str, help="Bandit method to use", default=None, choices=bandit_list)
args = parser.parse_args()

assert args.bandit_type is not None
bandit_type = args.bandit_type
if (len(args.params_path)>0):
    with open("results/"+args.params_path+"/params.json", "r") as f:
        params = json.load(f)
    for a in params:
        globals()[a] = params[a]

if (beta_type=="heuristic"):
    beta_type_ = "heuristic"
else:
    beta_type_ = "theoretical"

sb.call("mkdir -p data/", shell=True)

title="%s_K=%d_M=%d_N=%d_datatype=%s_ninstances=%d" % ("AI",K,M,N,data_type,ninstances)

data_params = {
    "data_type": data_type,
    "K": K,
    "M": M,
    "N": N,
    "alpha": alpha,
    "delta_min": delta_min,
}
instances = create_save_instances(ninstances, data_params, "_".join(title.split("_")[1:]), folder="data/")
title+="_alpha=%f_delta=%f_beta=%s" % (alpha,delta,beta_type_)
title_bandit=("bandit=%s_" % bandit_type)+title

W = instances[0]["W"]
mu = instances[0]["mu"]
K, M = mu.shape
## unstructured
A = np.matrix(np.eye(K))
S_star = instances[0].get("S_star",None)
gaps = instances[0].get("gaps", None)

print("Dataset: %s, Delta'_min = %.5f, alpha=%.2f, delta=%f"% (data_type,np.min(gaps),alpha,delta))

problem = eval("problems."+problem_type)(mu, sigma=sigma)

bandit_params = {
        "gaps": gaps, 
        "S_star": S_star, 
        "beta": eval("betas."+beta_type+"Threshold"), 
        "exploration": eval("betas."+exploration_type+"Exploration"),
        "alpha": alpha,
        "collabbeta": eval("betas."+collabbeta_type+"Threshold"),
        "delta": delta,
        "title": title_bandit,
}

bandit = eval("algorithms."+bandit_type)(bandit_params)

## PURE EXPLORATION
if ("AI" in bandit_type.split("_")[1]):
    if (action_type=="plot"):
        ## boxplot for experiment
        fnames = glob("results/"+title+"/*"+title+".csv")
        methods = [fn.split(title+"/")[1].split("bandit=")[1].split("_")[0] for fn in fnames]
        results_di = {methods[ifn]: pd.read_csv(fn, sep=",", index_col=0).T for ifn, fn in enumerate(fnames)}
        for method in results_di:
            boxplot({method: results_di[method]}, delta, folder="plots/"+title+"/", name=method)
        if (len(results_di)>0):
            boxplot(results_di, delta, folder="plots/"+title+"/")
        exit()

    if (njobs==1):
        SC, CT, RD, RT = bandit.run(W,A,problem,niter,seednb=seed_nb,max_samples=max_samples)
    else:
        seeds = [np.random.randint(int(1e8)) for _ in range(niter)]
        def single_run(jid, sd):
            return bandit.run(W,A,problem,1,jobid=jid,seednb=sd,max_samples=max_samples)
        results = Parallel(n_jobs=njobs, backend='loky')(delayed(single_run)(jid+1, sd) for jid, sd in enumerate(seeds))
        SC = [r[0][0] for r in results]
        CT = [r[1][0] for r in results]
        RD = [r[2][0] for r in results]
        RT = [r[3][0] for r in results]

    ## Save results
    sb.call("mkdir -p results/"+title+"/ plots/"+title+"/", shell=True)

    results = pd.DataFrame([SC, CT, RD, RT], index=["Samples", "Correctness", "Rounds", "Runtime"])
    results.to_csv("results/"+title+"/"+title_bandit+".csv")

    boxplot({bandit_type: results.T}, delta, folder="plots/"+title+"/", name=bandit_type.split("_")[0])

    print("Avg.Samples=%d\tAvg.Error=%.3f\tAvg.Rounds=%.3f" % (np.mean(SC),1-np.mean(CT),np.mean(RD)))

    ## boxplot for experiment
    fnames = glob("results/"+title+"/*"+title+".csv")
    methods = [fn.split(title+"/")[1].split("bandit=")[1].split("_")[0] for fn in fnames]
    results_di = {methods[ifn]: pd.read_csv(fn, sep=",", index_col=0).T for ifn, fn in enumerate(fnames)}
    if (len(results_di)>0):
        boxplot(results_di, delta, folder="plots/"+title+"/")

else:
    raise ValueError("Undefined problem/algorithm.")

arg_params = {a:eval(a) for a in ["K","M","N","alpha","data_type","beta_type","collabbeta_type","exploration_type","niter","delta"]}
with open("results/"+title+"/params.json", "w", encoding="utf-8") as f:
    json.dump(arg_params, f, ensure_ascii=False, indent=4)
