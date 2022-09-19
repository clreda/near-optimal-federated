#coding:utf-8

######################
## Parameters to modify

## Instance-relative parameters
K=6    #Number of arms
M=3 #Number of agents
N=1 #Number of optimal arms
alpha=0.5   #Personalization degree
data_type="personalizedsynthetic"   #Data type: see below

## Bandit-relative parameters
#bandit_type="FPE_AI"    #Type of bandit: set in command line
beta_type_="heuristic"   #Type of threshold for the confidence intervals
exploration_type="explog"   #Type of exploration/deterministic length

## Run-relative parameters
niter=100   #number of simulations
action_type="run"   #"run": run simulations, "plot": create a plot from prior results
delta=0.1   #error rate bound
parallel=True

######################
## Fixed parameters
from multiprocessing import cpu_count
njobs=max(1,cpu_count()-2) if (parallel) else 1
problem_type="Gaussian"
seed_nb=0
ninstances=1
max_samples=int(1e6)
delta_min=0.05  #Minimal gap between Nth and (N+1)th best arms in synthetic instances
## Fixed communication cost (cost per round of communication) is C=1
sigma=1
if (beta_type_ == "heuristic"):
    collabbeta_type = "heuristic"
    beta_type = "heuristic"
else:
    collabbeta_type = "alpha"
    beta_type = "mixture"

assert njobs < cpu_count()
assert problem_type in ["Gaussian"]
assert data_type in ["synthetic","personalizedsynthetic"]
assert collabbeta_type in ["elim", "alpha", "heuristic"]
assert beta_type in ["heuristic", "mixture"]
assert exploration_type in ["exp","log","explog"]
bandit_list = ["FPE_AI","ORACLE_AI", "PFLUCB_BAI"]
assert action_type in ["plot", "run"]

## Data_types:
#   -   synthetic: generation of a generic federated learning model
#   -   personalized synthetic: generation of a personalized federated model, with W = alpha*Id+(alpha+(1-alpha)/M)*1
