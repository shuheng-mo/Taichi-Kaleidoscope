from xxlimited import foo
import taichi as ti

"""Object-Oriented Programming in Taichi
@author: Shuheng Mo
@time: 2022-08-27 21:48:09
"""

# init
ti.init(default_ip=ti.i32)
ti.init(default_fp=ti.f64)
ti.init(ti.gpu)
# ti.init(ti.cpu,device_memory_fraction=0.5)
# ti.init(ti.cpu,device_memory_GB = 4.0)
ti.init(packed=True)

# OOP in Taichi is similar to Python, all we need is a decorator


@ti.data_oriented
class myWheel:
    def __init__(self, radius, width, rolling_fric) -> None:
        self.radius = radius
        self.width = width
        self.rolling_fric = rolling_fric
        
    @ti.func
    def foo(self):
        ti.static_print("foo")

    @ti.kernel
    def Roll(self):
        ti.static_print(self.radius) # ti.static_print(), faster run-time print
        ti.static_print(self.width)
        ti.static_print(self.rolling_fric)
        self.foo()

m = myWheel(radius=1000, width=1000, rolling_fric=0.985574)
m.Roll()
