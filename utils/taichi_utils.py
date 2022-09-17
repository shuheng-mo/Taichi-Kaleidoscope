import taichi as ti

"""
author: Shuheng Mo
time: 2022-09-02 15:49:34
"""


def initialization():
    """_summary_
    """
    ti.init(arch=ti.cpu)
    # ti.init(arch=ti.cpu, device_memory_fraction=0.8)
    # ti.init(arch=ti.gpu, device_memory_fraction=0.8)
    # ti.init(ti.cpu,device_memory_GB = 4.0)
    ti.init(default_ip=ti.i32)  # set up global precision for integers
    ti.init(default_fp=ti.f32)  # set up global precision for floats
    ti.init(packed=True)  # need data with shape 2^n
