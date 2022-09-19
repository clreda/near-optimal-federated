#coding: utf-8

import numpy as np
from random import sample

#' @param arr NumPy array
#' @param n integer
#' @param f function taking a list as argument
#' and returning a value of the same type as the elements of input list
#' @return n random indices i1,i2,...,in of the list which satisfy for all i in i1,i2,...,in, arr[i] == f(arr)
def randf(arr, n, f):
    return sample(np.array(np.argwhere(arr == f(arr))).flatten().tolist(), n)

def argmax_m_ls(ls, N):
    assert N <=len(ls)
    allowed = list(range(len(ls)))
    inds = [None]*N
    for i in range(N):
        idx = randf(np.array([ls[i] for i in allowed]), 1, np.max)[0]
        inds[i] = allowed[idx]
        del allowed[idx]
    return inds

#' @param mat NumPy matrix K x M with a total order natively implemented in Python
#' @param N integer
#' @returns returns M lists of N distinct indices and their values of the list which values are the N maximal ones (with multiplicity)
#'          argmax^N_{k <= K} mat[k,m] for each m <= M, max^N_{k <= K} mat[k,m] for each m <= M
def argmax_m(mat, N):
    K, M = mat.shape
    indices = [None]*M
    values = [None]*M
    for m in range(M):
        indices[m] = argmax_m_ls(mat[:,m].flatten().tolist()[0], N)
        values[m] = [mat[i,m] for i in indices[m]]
    return indices, values

def boxplot(results_di, delta, folder="", name="boxplot"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=(5,5), nrows=1, ncols=1)
    fm = str(len(str(delta))-2+1)
    fsize,markersize,rot=15,10,0 if (len(results_di)==1) else 27
    errors = [1-np.mean(results_di[method]["Correctness"]) for method in results_di]
    complexities = [np.mean(results_di[method]["Samples"]) for method in results_di]
    stds = [np.std(results_di[method]["Samples"]) for method in results_di]
    rounds = [np.mean(results_di[method]["Rounds"]) for method in results_di]
    stdrounds = [np.std(results_di[method]["Rounds"]) for method in results_di]
    idx = np.argsort(complexities).tolist()
    methods = [None]*len(results_di)
    for im, method in enumerate(results_di):
        methods[idx.index(im)] = method
    errors = [errors[i] for i in idx]
    complexities = [complexities[i] for i in idx]
    stds = [stds[i] for i in idx]
    rounds = [rounds[i] for i in idx]
    stdrounds = [stdrounds[i] for i in idx]
    if (len(results_di)<3):
        labels = [r''+method+(("\n$\hat{\delta}=%."+fm+"f") % errors[im])+("$\n$\hat{r}=%d,\hat{c}=%d$" % (rounds[im],complexities[im]) if (len(results_di)>1) else ("\leq" if (errors[im]<=delta) else ">")+(("\delta=%."+fm+"f") % delta)+"$\n$\hat{r}=%d \pm %d, \hat{c}=%d \pm %d}$" % (rounds[im], stdrounds[im], complexities[im], stds[im])) for im, method in enumerate(methods)]
    else:
        labels = [r''+method[:3]+((" $\hat{\delta}=%."+fm+"f") % errors[im])+(",\hat{r}=%d$" % (rounds[im])) for im, method in enumerate(methods)]
    medianprops = dict(linestyle='-', linewidth=2.5, color='lightcoral')
    meanpointprops = dict(marker='.', markerfacecolor='white', alpha=0.)
    meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black', markersize=markersize)
    bplot = sns.boxplot(data=[results_di[method]["Samples"] for method in methods], ax=ax, showmeans=False, color='skyblue')
    bplot = sns.stripplot(data=[results_di[method]["Samples"] for method in methods], jitter=True, marker='o', alpha=0.1, color="grey")
    ax.plot(ax.get_xticks(), complexities, "kD", label="mean", markersize=markersize)
    ax.set_xticklabels(labels, rotation=rot, fontsize=fsize, fontweight="bold")
    for ti, t in enumerate(ax.xaxis.get_ticklabels()):
        t.set_color('red' if (errors[ti] > delta) else 'black')
    ax.set_ylabel("Exploration cost", fontsize=fsize)
    plt.legend(fontsize=fsize)
    boxplot_file = folder+name+".png"
    plt.savefig(boxplot_file, bbox_inches="tight")
    plt.close()
    print("Saved to "+boxplot_file)
