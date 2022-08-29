import taichi as ti
import torch
# import torch.nn as nn
# import numpy as np

"""Explore how can Taichi work with other deep learning frameworks
@author: Shuheng Mo
@time: 2022-08-29 11:34:52
"""

# init
ti.init(ti.gpu)
ti.init(default_ip=ti.i32)
ti.init(default_fp=ti.f32)
ti.init(packed=True)

# 1D
n = torch.tensor([1.0, 2.0, 3.0])
a = ti.field(ti.f32, 3)
a.from_torch(n)


@ti.kernel
def work_with_torch_1D():
    for i in ti.grouped(a):
        print(a[i])
    # print('test')


work_with_torch_1D()
