from turtle import shape
import taichi as ti
from time import perf_counter

"""meta programming in Taichi
@author: Shuheng Mo
@time: 2022-08-27 14:24:15
"""

# ti.init(ti.gpu,device_memory_faction = 0.5)
ti.init(ti.cpu)

# more like template or type <T> in C++

# def copy_1d(src,dst,size):
#     for i in range(size):
#         dst[i] = src[i]

# you may pass anything ti.kernel would recognize
# of course, it is pass by REFERENCE, but note we cannot modify a var defined in Python scope
x = ti.field(dtype=ti.f64, shape=())
x[None] = 100.0


@ti.kernel
def change_global(x: ti.template()):
    x[None] += 1
    print(x[None])


@ti.kernel
def check_global(x: ti.template()):
    print(x[None])


change_global(x)
check_global(x)


v1 = ti.Vector([1.0, 2.0])
v2 = ti.Vector.field(n=3, dtype=ti.f32, shape=())
v3 = ti.Vector([0, 0])
v4 = ti.Vector.field(n=2, dtype=ti.f32, shape=())


@ti.kernel
def copy_1d(src: ti.template(), dst: ti.template(), size: ti.template()):
    src[0] = 1  # ok
    # v1[0] = 1 # err
    # v2[None] = [1,1,1] # ok for the ti.field
    for i in range(size):
        dst[i] = src[i]


a = ti.field(ti.f32, 4)
b = ti.field(ti.f32, 4)
c = ti.field(ti.f32, 42)
d = ti.field(ti.f32, 42)
err = [42, 3.14]

copy_1d(a, b, 4)
copy_1d(c, d, 42)
# copy_1d(err,d,42)
# copy_1d(v1,v3,2)
# copy_1d(v2,v4,2

# The best practice is to use struct-for to do the copy, we can use indices also


@ti.kernel
def copy_2d(x: ti.template(), y: ti.template()):
    for i, j in x:
        x[i, j] = [1, 2]

    for i, j in x:
        y[i, j] = x[i, j]


@ti.kernel
def copy_3d(x: ti.template(), y: ti.template()):
    # for i,j,k in x:
    #     x[i,j,k] = [3,4,5]

    # for i,j,k in x:
    #     y [i,j,k] = x[i,j,k]

    # use of ti.grouped(ti.field), this is the recommended practice
    # which get all the indices of the field
    for I in ti.grouped(x):
        x[I] = [3, 4, 5]

    for I in ti.grouped(x):
        y[I] = x[I]


mat1 = ti.Vector.field(n=2, dtype=ti.f32, shape=(2, 2))
mat2 = ti.Vector.field(n=2, dtype=ti.f32, shape=(2, 2))
mat3 = ti.Vector.field(n=3, dtype=ti.f32, shape=(2, 2, 2))
mat4 = ti.Vector.field(n=3, dtype=ti.f32, shape=(2, 2, 2))
copy_2d(mat1, mat2)
print(mat2)

copy_3d(mat3, mat4)
print(mat4)


l = ti.field(dtype=ti.f32, shape=(2, 2))


@ti.kernel
def get_indices(l: ti.template()):
    for I in ti.grouped(l):
        print(I)


get_indices(l)

# what is the n and m for vector and matrices? can we get them?


@ti.kernel
def get_n_dims():
    vec_test = ti.Vector([7, 8, 9])
    print(vec_test.n)  # rows: 3
    print(vec_test.m)  # cols: 1

    mat_test = ti.Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(mat_test.n)  # rows: 2
    print(mat_test.m)  # cols: 4

    # we can get the dim of field in the same way I think
    # what if higher dimensions ?s


get_n_dims()

# First glance at run-time optimization: ti.static()

# case 1: if we know some branch is certain (Not changed), we do not need branching in run-time
enable_run = False


@ti.kernel
def no_branch():
    if ti.static(enable_run):
        print("do something")


start = perf_counter()
no_branch()
end = perf_counter()
print(end - start)

# @ti.kernel
# def branch():
#     if enable_run:
#         print("do something")

# start = perf_counter()
# branch()
# end = perf_counter()
# print(end - start)

# no branch: 0.0028945831581950188, branch: 0.01150987483561039

# case 2: unroll the loops
# if we do not want to parallel some loop
# for i in ti.static(range(10)):

# case 3: sometimes we must unroll
# e.g we want to modify the first vector's first element
# not supported for index accessing [i][j] if j is not a constant
s = ti.Vector.field(3, ti.f32, shape=(8))


@ti.kernel
def reset():
    for i in s:
        # for j in range(3): # not working
        for j in ti.static(range(s.n)):
            # s[i][0] is ok
            s[i][j] = 8


reset()
print(s)
