from turtle import shape
import taichi as ti
import torch
import torch.nn as nn 

"""Taichi data visualisation
@author: Shuheng Mo
@time: 2022-08-28 13:37:10
"""

ti.init(ti.gpu,device_memory_fraction=0.5)

# note print on GPU is not the same as CPU, while calculation is
@ti.kernel
def print_gpu():
    for i in range(50):
        print(i) # not in order when on GPUs, could be random
    print("Inside the kernel")
        
print("kernel init")
print_gpu()
print("kernel dead")
ti.sync() # to make sure we output things in GPU, use ti.sync()
print("sync done")

# ti.GUI is slow and weak, 2D only, not the most optimal solution