import time
import taichi as ti
from utils import taichi_utils

# init
taichi_utils.initialization()

# sometimes more time on mem access but not computation
# data-access optimization

x = ti.field(ti.i32, shape=16)
y = ti.field(ti.i32, shape=(4, 4))


@ti.kernel
def fill_1D():
    for i in x:
        x[i] = i


@ti.kernel
def fill_2D():
    for i, j in y:
        y[i, j] = 10 * i + j


fill_1D()
print(x)

fill_2D()
print(y)

# we can access either in row-major,col-major or block-major
# but how to store it like we want to access in RAM?
# ideally, we can improve our field

# take 1D vector field as example
z = ti.Vector.field(3, ti.f32, shape=16)

# convert to
z = ti.Vector.field(3, ti.f32)
ti.root.dense(ti.i, 16).place(z)

# what is happening?
# SNode-tree is specified, then the row-major, column-major problem is solved, e.g.
a = ti.field(ti.i32, shape=(4, 4))

# to specify row-major (is row-major by default)
# ti.root.dense(ti.ij, (4, 4)).place(a)

# col-major
# ti.root.dense(ti.jk,(4,4)).place(a)

# or equivalent as
# ti.root.dense(ti.i,4).dense(ti.j,4).place(x)

# Another best practice this case, we use Hierachical field
b = ti.field(ti.i32)
ti.root.dense(ti.i, 4).dense(ti.i, 4).place(b)


@ti.kernel
def print_field_1D():
    for i in b:
        b[i] = i
        print(b[i], end=' ')


print_field_1D()

# what is so good about this is, we could form a field in higher dimension but access it in 1D manner


# how about block majored access? how to do it
c = ti.field(ti.i32)
ti.root.dense(ti.ij, (2, 2)).dense(ti.ij, (2, 2)).place(c)


@ti.kernel
def print_blocks():
    print('The indices for block-majored access:')
    for i in ti.grouped(c):
        print(i, end=' ')


print_blocks()

# AOS and SOA. Which is better? no, it really depends what data are you accessing
arr_1 = ti.field(ti.i32)
arr_2 = ti.field(ti.i32)

# SOA
# ti.root.dense(ti.i,8).place(arr_1)
# ti.root.dense(ti.i,8).place(arr_2)

# AOS
ti.root.dense(ti.i, 8).place(arr_1, arr_2)


@ti.kernel
def access_merged():
    print("Accessing SOA: ")
    for i in ti.grouped(arr_1):
        print(i, end=' ')

    for i in ti.grouped(arr_2):
        print(i, end=' ')


start = time.process_time()
access_merged()
end = time.process_time()
print('Access time: ', end - start)

# can you explain why is this?
# 0.036502000000000034 SOA
# 0.051219000000000126 AOS

# to see the diffs, we can try on a physics problem:
h = 1000
substepping = 100
m = 2
N = 10000000
# pos = ti.Vector.field(2, ti.f32, N)
# vel = ti.Vector.field(2, ti.f32, N)
# force = ti.Vector.field(2, ti.f32, N)

# AOS
pos = ti.Vector.field(4, ti.f32)
vel = ti.Vector.field(4, ti.f32)
force = ti.Vector.field(4, ti.f32)
ti.root.dense(ti.i, N).place(pos, vel, force)


@ti.kernel
def update():
    dt = h/substepping
    for i in range(N):
        vel[i] += dt*force[i]/m
        pos[i] += dt*vel[i]


start = time.process_time()
update()
end = time.process_time()

# 0.07253300000000018 -> 0.07627399999999995
print('Update time: ', end - start)
