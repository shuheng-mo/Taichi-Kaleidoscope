'''
 # @ Author: Shuheng Mo
 # @ Create Time: 2022-09-22 19:01:58
 # @ Modified by: Shuheng Mo
 # @ Modified time: 2022-09-25 22:20:17
 # @ Description: Common functions that often used in taichi
 '''

import taichi as ti


def initialization():
    """_summary_
    """
    ti.init(arch=ti.cpu)
    # ti.init(arch=ti.cpu, device_memory_fraction=0.8)
    # ti.init(arch=ti.gpu, device_memory_fraction=0.8)
    # ti.init(ti.cpu,device_memory_GB = 4.0)
    ti.init(default_ip=ti.i32)  # set up global precision for integers
    ti.init(default_fp=ti.f32)  # set up global precision for floats
    # use double precision
    # ti.init(default_ip=ti.i64)  # set up global precision for integers
    # ti.init(default_fp=ti.f64)  # set up global precision for floats
    ti.init(packed=True)  # need data with shape 2^n
