import taichi as ti

"""Taichi Kernels and functions.
@author: Shuheng Mo
@time: 2022-08-28 13:37:10
"""

# init
ti.init(ti.cpu)

# how does global values works with taichi scopes?
global_val = 1999


@ti.func
def summation(x: ti.i32, y: ti.f32) -> ti.f32:
    # must specify the arg type when pass in to func
    return x + y


@ti.kernel
def round(x: ti.i32, y: ti.f32) -> ti.i32:
    return summation(x, y)


print(round(24, 39.77))  # round lower

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
    a = test_kernel()  # to show this result, we move it to a ti.func
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

# What if I the code I want to parallelize is not optimal? like:
# @ti.kernel
# def my_loop():
#     for i in range(10): # this is fine
#         for j in range(100): # this is the dominant
#             ...

# @ti.kernel
# def bigger_loops():
#     for j in range(100): # this get parallelised
#         ...

# # Then we need carefully design the algorithm, like
# def smaller_loops():
#     for _ in range(10):
#         bigger_loops()

# is this really working?

# The best practice for atomic operation in the parallel loops
total = ti.field(dtype=ti.float64, shape=())


@ti.kernel
def parallel_sum():
    for i in range(100):
        # good pratices
        total[None] += 1  # atomic, safe

        ti.atomic_add(total[None], 1)  # atomic, safe

        # total[None] = total[None] + 1 # possible, not safe and nor atomic, race condition


parallel_sum()
print(total)  # 200

# advanced topics

# For things under Taichi scope, it's all static
# we have static data type, static lexical scope


@ti.kernel
def err_not_static(x: float):
    a = 100
    a = 99.77  # implicit conversion, we have 99
    # a = ti.Vector([1,2,3]) # we do not have implicit convertion for this, thus static data type
    if x > 0:
        y = x
        print(x)
    else:
        y = -x
        print(y)
    # print(y) # y is out of scope, we cannot see it here, thus static lexical scope


err_not_static(-10)
