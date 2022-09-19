#coding: utf-8

import numpy as np
import os
import pickle
import pandas as pd
from params import seed_nb
import pickle

from utils import argmax_m

def compute_gaps(mu_p, N):
    K, M = mu_p.shape
    S_star, mu_star = argmax_m(mu_p, N+1)
    S_star = [S_star[m][:-1] for m in range(M)]
    mu_m = [mu_star[m][-2] for m in range(M)]
    mu_m1 = [mu_star[m][-1] for m in range(M)]
    gaps = np.matrix(np.zeros((K,M)))
    for m in range(M):
        for k in range(K):
            if (k in S_star[m]):
                gaps[k,m] = mu_p[k,m]-mu_m1[m]
            else:
                gaps[k,m] = mu_m[m]-mu_p[k,m]
    return gaps, S_star

def create_save_instances(ninstances,data_params,name,folder=""):
    data_type, K, M, N_ = [data_params[a] for a in ["data_type", "K","M","N"]]
    instances_fname = folder+name+".pck"
    if (not os.path.exists(instances_fname)):
        instances = {}
        for nid in range(ninstances):
            W, mu, S_star, gaps = eval(data_type+"_instance")(K,M,N_,data_params["delta_min"],data_params["alpha"])
            K, M = mu.shape
            instances.setdefault(nid, {'W': W, 'S_star': S_star, "mu": mu, "gaps": gaps})
        with open(instances_fname, "wb") as f:
            pickle.dump(instances, f)
    with open(instances_fname, "rb") as f:
        instances = pickle.load(f)
    alpha = data_params["alpha"]
    if ((alpha >= 0) and (data_type in ["personalizedsynthetic"])):
        for nid,NI in enumerate(instances):
            mu = instances[NI]["mu"]
            M = mu.shape[1]
            W = alpha*np.matrix(np.eye(M))+((1-alpha)/M)*np.matrix(np.ones((M,M)))
            gaps, S_star = compute_gaps(mu.dot(W), N_)
            instances[NI] = {"W":W, "S_star": S_star, "mu": mu, "gaps": gaps}
    return instances

def personalizedsynthetic_instance(K,M,N,delta_min,alpha):
    ## Weight matrix
    if (M==1):
        W = np.matrix([[1.]])
    else:
        W = np.matrix(((1-alpha)/M)*np.ones((M,M))+alpha*np.eye(M))
    assert (np.isclose(np.sum(W,axis=0),1)).all()
    while True:
        mu = np.matrix(np.random.normal(0,1,(K,M))).reshape((K,M))
        mu /= np.linalg.norm(mu,None)
        gaps, S_star = compute_gaps(mu.dot(W), N)
        delta_min_ = np.min(gaps)
        if ((delta_min_>=delta_min) and (delta_min_>0)):
            break
    return W, mu, S_star, gaps

#' @param K number of arms
#' @param M number of agents
#' @param delta_min minimum value of gap
#' @param alpha personalization degree
#' @return W, mu, k_star, gaps
def synthetic_instance(K,M,N,delta_min,alpha=None):
    ## Weight matrix
    if (M==1):
        W = np.matrix([[1.]])
    else:
        alpha_ = np.random.choice([i*0.1 for i in range(0, 11, 1)],M)
        W = np.matrix(np.ones((M,M)))
        for m in range(M):
            W[:,m] = (1-alpha_[m])/M*W[:,m]
        np.fill_diagonal(W,alpha_+(1-alpha_)/M)
    assert (np.isclose(np.sum(W,axis=0),1)).all()
    while True:
        mu = np.matrix(np.random.normal(0,1,(K,M))).reshape((K,M))
        mu /= np.linalg.norm(mu,None)
        mu_p = mu.dot(W)
        gaps, S_star = compute_gaps(mu_p, N)
        delta_min_ = np.min(gaps)
        if ((delta_min_>=delta_min) and (delta_min_>0)):
            break
    return W, mu, S_star, gaps
