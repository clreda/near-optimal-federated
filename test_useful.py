# coding:utf-8

from solving import solve_oracle
import numpy as np

type_ = "synthetic"

if (type_ == "synthetic"):
    M = K = 2
    #W = 1/M*np.matrix(np.ones((M,M)))
    sim = 0.9
    W = np.matrix([[1,sim],[sim,1]])/(1+sim)
    Id = np.matrix(np.eye(M))

    mu = np.matrix([[1,0.5],[0,0.1]])
    print("mu=\n%s" % (str(mu)))
    print("W=\n%s" % (str(W)))
    #mu = np.matrix([[1,0.3],[0.2,0.5]])

print(type_)

#N=1
res = []
for w in [W, Id]:
    mup = mu.dot(w)
    S_star = [[np.argmax(mup[:,m])] for m in range(M)]
    min_gaps = [mup[S_star[m][0],m]-np.max([mup[i,m] for i in range(K) if (i not in S_star[m])]) for m in range(M)]
    Delta = np.zeros((K,M))
    for m in range(M):
        Delta[:,m] = np.ravel(mup[S_star[m][0],m]-mup[:,m])
        Delta[S_star[m][0],m] = min_gaps[m]
    c = np.power(Delta, 2)/2
    _, v = solve_oracle(w, c, S_star)
    res.append(v)
v1, v1_Id = res

print("Value for W:\t%f" % v1)
print("Value for Id:\t%f" % v1_Id)
