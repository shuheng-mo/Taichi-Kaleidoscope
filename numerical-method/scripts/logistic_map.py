'''
 # @ Author: Shuheng Mo
 # @ Create Time: 2022-09-26 11:44:02
 # @ Modified by: Shuheng Mo
 # @ Modified time: 2022-09-26 11:45:07
 # @ Description:
 '''


import taichi as ti
from utils import taichi_utils
import numpy as np

# init
taichi_utils.initialization()

# 1D Linear stability analysis
# fixed point: x* = f(x*)
# non-fixed point: x1 = f(x0), x2 = f(x1)

# Xn+1 = mu * Xn * (1 - Xn)

mu_min = 2.4
mu_max = 4.0
n_mu = 500
n_x = 400
mu_edges = np.linspace(mu_min,mu_max,n_mu+1)
mu = (mu_edges[1:n_mu] + mu_edges[2:n_mu+1])/2

n_trans = 200000
n_data = 100000

x_data = np.zeros((n_data,n_mu))
print(x_data.shape)

# set up initial condition
x_0 = 0.5

# TODO: the main logic of the logistic map