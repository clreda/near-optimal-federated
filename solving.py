# coding:utf-8

import pandas as pd
import numpy as np
import cvxpy as cp
from copy import deepcopy as dp
import random
import gc

## Parameters
accepted_status = ["optimal","optimal_inaccurate"]
eps=lambda a : a+2*np.finfo(np.float32).eps
solver_params = {}

## Utils
def problem_solution_util(pb, verbose, solver=cp.MOSEK, alt_solver=cp.ECOS, qcq=False):
    try:
        pb.solve(verbose=verbose,qcp=qcq,solver=solver,**solver_params)
    except cp.error.SolverError:
        if (verbose):
            print("solver %s failed to find allocation" % solver)
        if (solver == "MOSEK"):
            allocation, value = problem_solution_util(pb, verbose, solver=alt_solver, alt_solver=cp.SCS)
            return allocation, value
        elif (solver == "ECOS"):
            allocation, value = problem_solution_util(pb, verbose, solver=alt_solver)
            return allocation, value
        raise ValueError("solver failed to find allocation")
        return None
    if (verbose):
        print("------------- Problem status: %s -----------------" % pb.status)
    if (pb.status in accepted_status):
        allocation = np.matrix(list(pb.solution.primal_vars.values())[0])
        return allocation, pb.value
    else:
        print("optimization problem "+str(pb.status))
        if (verbose):
            try:
                pb.solve(verbose=True,qcq=qcq,solve="MOSEK",**solver_params)
            except:
                pass
        print(pb)
        raise ValueError("optimization problem "+str(pb.status))
        return None

#---------------------------------------
# SOLVING OPTIMIZATION PROBLEMS RELATED TO THE LOWER BOUND

#' @param W
#' @param c
#' @param S_star
#' @param N
#' @param verbose
#' @return the allocation and the value solution to the oracle problem, that is,
#'          \min_t \sum_{k,m} t_{k,m}
#'          s.t.    for all m, k not in S^\star_m, l in S^\star_m, \sum_n w_{n,m}^2(1/t_{k,n}+1/t_{l,m}) <= (\mu'_{k,m}-\mu'_{l,m})^2/2
def solve_oracle(W, c, S_star, N=1, verbose=False):
    K, M = c.shape
    taus = cp.Variable((K, M))
    objective = cp.Minimize(cp.sum(taus))
    W_2 = np.power(W, 2)
    constraints = [W_2[:,m].T @ (cp.inv_pos(taus[k,:])+cp.inv_pos(taus[l,:])) <= c[k,m] for m in range(M) for k in range(K) if (k not in S_star[m]) for l in S_star[m]]
    pb = cp.Problem(objective, constraints)
    allocation, value = problem_solution_util(pb, verbose)
    gc.collect()
    return allocation, value

#' @param k
#' @param W
#' @param c
#' @param verbose
#' @return the allocation solution to the decoupled BAI relaxed problem for a fixed arm k, that is,
#'          \min_t \sum_{m} t_{k,m}
#'          s.t.    for all m, \sum_{n} w_{n,m}^2/t_{k,n} <= c_{k,m}
def solve_decoupled(k, W, c, verbose=False):
    _, M = c.shape
    if (verbose):
        print("k=%d\tr=%d" % (k,np.round(np.max(0.5*np.log(1/c)/np.log(2)))))
        print("c=\n%s\n" % str(c))
    tau = cp.Variable((1,M))
    objective = cp.Minimize(cp.sum(tau))
    W_2 = np.power(W, 2)
    constraints = [W_2[:,m].T @ cp.inv_pos(tau[0,:]) <= c[k,m] for m in range(M)]
    pb = cp.Problem(objective, constraints)
    tau_, _ = problem_solution_util(pb,verbose)
    gc.collect()
    return tau_

#' @param W
#' @param c
#' @param verbose
#' @return the allocation and the value solution to the BAI relaxed problem, that is,
#'          \min_ \sum_{k,m} t_{k,m}
#'          s.t.    for all m,k, \sum_{n} w_{n,m}^2/t_{k,n} <= c_{k,m}
def solve_relaxed(W, c, verbose=False):
    K, M = c.shape
    taus = [solve_decoupled(k, W, c, verbose) for k in range(K)]
    allocation = np.matrix(np.concatenate(taus, axis=0))
    value = np.sum(allocation)
    if (verbose):
        print("-"*10)
    return allocation, value

#_______________________________________
# SOLVING OPTIMIZATION PROBLEMS RELATED TO ALLOCATION TRACKING

from scipy.optimize import minimize, Bounds, NonlinearConstraint
def solve_allocation_decoupled(t_k, beta, cum_sum, verbose=False):
    _, M = t_k.shape
    d0 = np.array([1]*M)
    objective = lambda d : np.sum(d)
    def cons_f(d):
        return [(cum_sum[0,m]+d[m])/beta(cum_sum[0,:]+d)-t_k[0,m] for m in range(M)]
    constraints = NonlinearConstraint(fun=cons_f,lb=0,ub=np.inf)
    tol = 2*eps(0)
    bounds = Bounds(0,float("inf"))
    options = {'maxiter':1000,"disp":verbose,"verbose":0,"gtol":tol,"xtol":tol}
    res = minimize(objective, d0, args=(), method="trust-constr", options=options, constraints=constraints, bounds=bounds, tol=tol)
    d = np.matrix(res.x).reshape((1,M))
    d[d<0]=0
    gc.collect()
    return d

#' @param t matrix of shape (K,M)
#' @param beta function of R^N into R
#' @param cum_sum integer matrix of shape (K,M)
#' @param verbose
#' @return the allocation and the value solution to 
#'          \min_n \sum_{k,m} d_{k,m}
#'          s.t.    for all m,k, (n_{k,m}(r-1)+d_{k,m})/(beta(n_{k,.}(r)+d_{k,.})) >= t_{k,m}
def solve_allocation(t, beta, cum_sum, verbose=False):
    K, M = t.shape
    Ds = [solve_allocation_decoupled(t[k,:], beta, cum_sum[k,:], verbose=verbose) for k in range(K)]
    allocation = np.matrix(np.concatenate(Ds, axis=0))
    value = np.sum(allocation)
    return np.ceil(allocation), value
