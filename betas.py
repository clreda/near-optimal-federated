#coding:utf-8

from scipy.special import zeta
from scipy.optimize import minimize_scalar
import numpy as np

eps=lambda a : a+2*np.finfo(np.float32).eps

## REGRET

def explogExploration(T,A,W=None,verbose=False):
    return lambda p : 2**p*np.log(T)

def expExploration(T,A,W=None,verbose=False):
    return lambda p : 2**p

def logExploration(T,A,W=None,verbose=False):
    return lambda p : 10*np.log(T)

## PURE EXPLORATION

def heuristicThreshold(delta, A, W=None, verbose=False):
    return lambda N : np.log((1+np.log(np.sum(N)))/float(delta))

## [Kaufmann and Koolen, 2021, JMLR]
#' @param delta
#' @param A
#' @returns function of R^N into R
#' /!\ works for Gaussian distributions (probably also for subGaussian distributions)
def mixtureThreshold(delta, A, W=None, verbose=False):
    K = A.shape[1]
    if (W is None):
        M = 1
    else:
        M = W.shape[1]
    ## corollary 10
    def gG(lbd):
        assert (lbd <= 1) and (lbd > 1/2)
        return 2*lbd-2*lbd*np.log(4*lbd)+np.log(zeta(2*lbd))-0.5*np.log(1-lbd)
    ## definition 3
    def Cg(x):
        assert x > 0
        lbd0 = 1
        obj = lambda l : (gG(l)+x)/l
        bounds=(1/2, 1)
        res = minimize_scalar(obj, lbd0, args=(), method="bounded", bounds=bounds)
        assert res.success
        return obj(res.x)
    ## combine with union bound on [K]x[M]
    gM = M*Cg(np.log((K*M)/delta)/M)
    return lambda N : gM+2*np.sum(np.log(4+np.log(N)))

def elimThreshold(delta, A, W=None, verbose=False):
    K = A.shape[1]
    if (W is not None):
        K *= W.shape[1]
    return (lambda t : np.log(4*K*(t**2)/delta))

def alphaThreshold(delta, A, W=None, alf=2, verbose=False):
    K = A.shape[1]
    if (W is not None):
        K *= W.shape[1]
    z_alpha = zeta(alf)
    return (lambda t : np.log(K*z_alpha*(t**alf)/delta))
