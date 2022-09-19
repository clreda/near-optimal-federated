#coding:utf-8

from solving import solve_oracle, solve_relaxed, solve_allocation
from utils import argmax_m
from betas import alphaThreshold, heuristicThreshold, elimThreshold

import numpy as np
from random import seed as randomseed
from time import time
import sys
sys.path.insert(0, "PFMAB/")
import subprocess as sb
import pickle

## FEDERATED PURE EXPLORATION

class FederatedAI(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.pulls = None
        self.rewards = None
        self.cumsum = None

    def decide(self, verbose=False):
        raise NotImplemented

    #' @param W weight matrix
    #' @param A feature matrix
    #' @param problem
    #' @param niter 
    #' @param mu only to compute error frequency
    def run(self,W,A,problem,niter,N=1,jobid=None,seednb=0,max_samples=int(1e6),verbose=False):
        np.random.seed(seednb)
        randomseed(seednb)
        M = W.shape[1]
        ## sample complexity
        SC = np.array([None]*niter)
        ## error frequency
        CT = np.array([None]*niter)
        ## round complexity
        RD = np.array([None]*niter)
        ## runtime
        RT = np.array([None]*niter)
        ## not terminated runs
        NT = 0
        for nit in range(niter):
            start = time()
            Ans, Sc, Rd, nt = self.decide(W,A,problem,N=1,max_samples=max_samples,verbose=verbose)
            RT[nit] = time()-start
            SC[nit] = Sc
            RD[nit] = Rd
            NT += nt
            assert len(Ans)==M
            if (problem.mu is None):
                CT[nit] = np.nan
            elif (not all([len(x)==N for x in Ans])):
                CT[nit] = 0
            else:
                Mu_m = [[ls[-1]] for ls in argmax_m(problem.mu.dot(W), N)[1]]
                Mu_ans = [[problem.mu.dot(W)[aa,m] for aa in a] for m, a in enumerate(Ans)]
                CT[nit] = int(all([all([mu_ans >= Mu_m[m] for mu_ans in Mu_ans[m]]) for m in range(M)]))
            print("It#%d/%d\tSamples=%d\tStatus=%s\tRounds=%d" % (nit+1 if (jobid is None) else jobid,niter,SC[nit],"correct" if(CT[nit]==1) else "ERROR",RD[nit]))
        if (NT > 0):
            print("Not terminated runs %d" % NT)
        return SC, CT, RD, RT

## Translation of the regret-minimizing algorithm in [Shi et al., 2021] into a BAI
## /!\ works only for personalized federated instances
class PFLUCB_BAI(FederatedAI):
    def __init__(self,params):
        self.title = params["title"]
        self.beta_ = lambda A, W : params["collabbeta"](params["delta"], A, W)
        self.exploration_ = lambda A, W : params["exploration"](1/params["delta"], A, W)
        self.clear()

    def decide(self, W, A, problem, N=1, verbose=False, max_samples=None):
        assert N==1
        K, M = A.shape[1], W.shape[1]
        self.beta = self.beta_(A, W)
        self.exploration = self.exploration_(A,W)
        self.pulls = np.matrix(np.zeros((K,M)))
        self.rewards = {(k,m):[] for m in range(M) for k in range(K)}
        self.cumsum = np.matrix(np.zeros((K,M)))
        answers = [[] for m in range(M)]
        ## W = alpha*Id_M+(1-alpha)/M*1_M
        alpha = W[0,0]-W[0,1]
        p = 1
        Fp = 0
        ## Total number of samples (~ np.sum(self.pulls))
        nsamples = int(np.sum(self.pulls))
        ## Arms that are still active
        B = np.matrix(np.ones((K,M)))
        while ((max_samples is None) or (nsamples<max_samples)):
            fp = self.exploration(p)
            Fp += fp
            B_p = np.argwhere(np.sum(B,axis=1)>0)[:,0]
            d_p = np.matrix(np.zeros((K,M)))
            ## Global exploration for all players
            nglobal = np.ceil((1-alpha)*fp)
            ## Local exploration for all players and their active arms
            nlocal = np.ceil(alpha*M*fp)
            for m in range(M):
                for k in B_p:
                    B_m = np.argwhere(B[:,m]!=0)[:,0].tolist()
                    d_p[k,m] = nglobal+nlocal*int(k in B_m)
            d_p = d_p.astype(int)
            ids = np.argwhere(d_p!=0).tolist()
            for (k,m) in ids:
                if (nsamples+d_p[k,m]>=max_samples):
                    ndraw_s = max_samples-nsamples
                else:
                    ndraw_s = d_p[k,m]
                nsamples += ndraw_s
                rewards_list = problem.sample(k,m,ndraw_s)
                self.cumsum[k,m] += sum(rewards_list)
                self.rewards[(k,m)] = self.rewards[(k,m)]+rewards_list.tolist()
                self.pulls[k,m] += ndraw_s
                if (nsamples>=max_samples):
                    break
            means = self.cumsum/(self.pulls+int((self.pulls==0).any()))
            ## Server computes the average means
            avg_means = means.mean(axis=1)
            ## Players compute the mixed means
            fed_means = alpha*means+(1-alpha)*np.repeat(avg_means, M, axis=1)
            Bp = np.sqrt(2*self.beta(p)/(M*Fp))
            ## Elimination
            for m in range(M):
                B_m = np.argwhere(B[:,m]!=0)[:,0]
                if (len(B_m)>0):
                    max_mean = argmax_m(fed_means[B_m,m], N)[1][0][-1]
                    B[B_m,m] = (max_mean-means[B_m,m]<=2*Bp).T.astype(int)
                    B_m = np.argwhere(B[:,m]!=0)[:,0].tolist()
                    if (len(B_m)==1):
                        answers[m] = B_m
                        B[:,m] = 0
            p += 1
            if ((np.sum(B, axis=0)==0).all()):
                break
        nt = 1-int((np.sum(B,axis=0)==0).all())
        return answers, nsamples, p, nt

class FPE_AI(FederatedAI):
    def __init__(self,params):
        self.title = params["title"]
        self.beta_ = lambda A, W : (lambda N : 2*params["beta"](params["delta"], A, W)(N))
        self.clear()

    def omega(self, W, n):
        K, M = n.shape
        Omega = np.matrix(np.zeros((K,M)))
        W_2 = np.power(W.T, 2)
        invT = lambda k : np.power(n[k,:].T,-1)
        for k in range(K):
            Omega[k,:] = np.sqrt(self.beta(self.pulls[k,:])*np.power(W,2).T.dot(np.power(n[k,:], -1).T)).T
        return Omega

    def decide(self, W, A, problem, N=1, verbose=False, max_samples=None):
        self.clear()
        K, M = A.shape[1], W.shape[1]
        self.beta = self.beta_(A,W)
        ## self.pulls[k,m] = nkm(r) at round r
        self.pulls = np.matrix(np.ones((K,M)))
        ## Initialization
        self.rewards = {(k,m):[problem.sample(k,m,1)] for m in range(M) for k in range(K)}
        self.cumsum = np.matrix(np.zeros((K,M)))
        ## Total number of samples (~np.sum(self.pulls))
        nsamples = 0
        for m in range(M):
            for k in range(K):
                if (nsamples>=max_samples):
                    break
                nsamples += 1
                self.cumsum[k,m] = self.rewards[(k,m)][0]
        Delta_t=np.matrix(np.ones((K,M)))
        ## B[k,m] = int(k in Bm(r)) at round r
        B=np.matrix(np.ones((K,M)))
        r=0
        while ((max_samples is None) or (nsamples<max_samples)):
            if (r % 10 == 0 and r > 0):
                print("r=%d\t|B(r)|=%d" % (r, np.sum(B)))
            ## List of arms which are present in at least one Bm(r) set
            B_r = np.argwhere(np.sum(B,axis=1)>0)[:,0]
            t_r, _ = solve_relaxed(W, np.power(Delta_t,2))
            d_rt, _ = solve_allocation(t_r, self.beta, self.pulls, verbose=verbose)
            d_rt = d_rt.astype(int)
            ## Get empirical means
            idx = np.argwhere(d_rt[B_r,:]>0).tolist()
            d_r = np.max(d_rt[B_r,:].sum(axis=0))
            for (q,m) in idx:
                k = B_r[q]
                if (nsamples+d_rt[k,m]>=max_samples):
                    ndraw_s = max_samples-nsamples
                else:
                    ndraw_s = d_rt[k,m]
                nsamples += ndraw_s
                rewards_list = problem.sample(k,m,ndraw_s)
                self.cumsum[k,m] += sum(rewards_list)
                self.rewards[(k,m)] = self.rewards[(k,m)]+rewards_list.tolist()
                self.pulls[k,m] += ndraw_s
                if (nsamples>=max_samples):
                    break
            means = self.cumsum/(self.pulls+int((self.pulls==0).any()))
            means = means.dot(W)
            ## Elimination
            Omega = self.omega(W, self.pulls)
            for m in range(M):
                B_m = np.argwhere(B[:,m]!=0)[:,0]
                max_LCB_m = argmax_m(means[B_m,m]-Omega[B_m,m], N)[1][0][-1]
                B[B_m,m] = (means[B_m,m]+Omega[B_m,m]>=max_LCB_m).T.astype(int)
                if (np.sum(B[:,m])>N):
                    B_m = np.argwhere(B[:,m]!=0)[:,0]
                    Delta_t[B_m,m] = Delta_t[B_m,m].flatten()*0.5
            r += 1
            if ((np.sum(B, axis=0)<=N).all()):
                break
        Bm_s = [np.argwhere(B[:,m]!=0)[:,0].tolist() for m in range(M)]
        return Bm_s, nsamples, r, 1-int((np.sum(B,axis=0)<=N).all())

class ORACLE_AI(FederatedAI):
    def __init__(self,params):
        self.title = params["title"]
        self.gaps = params["gaps"]
        self.S_star = params["S_star"]
        self.delta = params["delta"]
        self.clear()

    def decide(self,W,A,problem, N=1, forced_exploration=False, max_samples=None, verbose=False):
        self.clear()
        K, M = A.shape[1], W.shape[1]
        T, _ = solve_oracle(W, np.power(self.gaps, 2)/2, self.S_star)
        self.pulls = np.ceil(np.log(1/(2.4*self.delta))*T).astype(int)
        self.rewards = {}
        self.cumsum = np.matrix(np.zeros((K,M)))
        nsamples = 0
        for (k,m) in np.argwhere(self.pulls>0).tolist():
            n_draws = int(self.pulls[k,m] if (not forced_exploration) else max(self.pulls[k,m], np.sqrt(K)))
            if (nsamples+n_draws>=max_samples):
                ndraw_s = max_samples-nsamples
            else:
                ndraw_s = n_draws
            nsamples += ndraw_s
            rewards_list = problem.sample(k,m,ndraw_s)
            self.rewards.setdefault((k,m), rewards_list)
            self.cumsum[k,m] += sum(self.rewards[(k,m)])
            self.pulls[k,m] = ndraw_s
            if (nsamples>=max_samples):
                break
        means = self.cumsum/(self.pulls+int((self.pulls).any()))
        means = means.dot(W)
        indices, _ = argmax_m(means, N)
        return indices, nsamples, 1, 0
