#coding:utf-8

import numpy as np

class Problem(object):
    def __init__(self):
        pass

class Gaussian(object):
    def __init__(self, mu, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def sample(self, arm, agent, n=1):
        return np.random.normal(self.mu[arm,agent], self.sigma, size=n)
