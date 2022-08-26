import taichi as ti

# init
ti.init(ti.cpu)

# how does global values works with taichi scopes?
global_val = 1999

@ti.func
def summation(x:ti.i32, y:ti.f32)->ti.f32:
    # must specify the arg type when pass in to func
    return x + y
    
@ti.kernel
def round(x:ti.i32,y:ti.f32) -> ti.i32:
    return summation(x,y)

print(round(24,39.77)) # round lower

# only ONE return value for taichi is allowed !!!

# @ti.kernel
# def error_multiple_return() -> (ti.i32, ti.f32):
#     x = 1
#     y = 0.5
#     return x, y  # Compilation error: more than one return value

# error_multiple_return() # return multi args not valid for ti.kernel

# @ti.kernel
# def test_sign(x: float) -> float:
#     if x >= 0:
#         return 1.0
#     else:
#         return -1.0
#     # Error: multiple return statements
    
# test_sign(-100) # multi returns for one ti.kernel does not make sense either

# N = 6
N = 5
matN = ti.types.matrix(N, N, ti.i32)
# matN = ti.field(ti.f32,(10,10))

@ti.func
def test_kernel() -> matN:
    return matN([[N] * N for _ in range(N)])
    # return matN
    # Compilation error: The number of elements is 36 > 30 for ti.kernel
    
@ti.kernel
def call_test_kernel():
    a = test_kernel() # to show this result, we move it to a ti.func
    print(a)
    
# a = test_kernel() # for ti.func we can return as elems as we want ...
# print(a)
call_test_kernel()    

@ti.kernel
def global_1():
    print(global_val)
    
@ti.kernel
def global_2():
    print(global_val)
    
# What will be the values?
global_1()
global_val = 2999
global_1()
global_2() 

# Note: a kernel does not track change of global vals after it has been compiled and executed

