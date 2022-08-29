import taichi as ti

"""Data structures used in Taichi
@author: Shuheng Mo 
@time: 2022-08-27 22:47:42
"""

ti.init(ti.gpu)
ti.init(default_ip=ti.i32)  # define the default data type
ti.init(default_fp=ti.f32)


@ti.kernel
def implicit_conversion():
    a = 100
    a = 127.7  # only commit the precision of first declare
    print(a)  # will be warned of precision loss


implicit_conversion()


@ti.kernel
def explicit_conversion():
    a = 100.06
    b = ti.cast(a, ti.i32)
    c = ti.cast(a, ti.f32)
    print(b)
    print(c)


explicit_conversion()

# to create data containers
# approach 1: define by ti.types before the kernel
# this is more like a meta programming way, not so recommended actually
vec3f = ti.types.vector(3, ti.f32)  # 1D array
mat2f = ti.types.matrix(2, 2, ti.f32)  # 2D array
ray = ti.types.struct(ro=vec3f, rd=vec3f, l=ti.f32)


@ti.kernel
def basic_data_structure():
    v1 = vec3f(12.0)
    v2 = vec3f(12.0, 13.0, 14.0)
    # approach 2: we can use predefined keys to do this (both kernel and python scope)
    v3 = ti.Vector([0, 2, 4])
    m1 = mat2f([1.3, 1.4], [1.5, 1.6])
    m2 = ti.Matrix([[3, 4], [5, 6]])
    r = ray(ro=v1, rd=v2, l=1)
    t = ti.Struct(v1=v3, v2=v3, l=1)
    print(v1)
    print(v2)
    print(v3)
    print(m1)
    print(m2)
    # print(r) # cannot print a struct directly
    print("r.ro = ", r.ro)
    print("r.rd = ", r.rd)
    print("r.l = ", r.l)
    print("t.v1 = ", t.v1)
    print("t.v2 = ", t.v2)
    print("t.l = ", t.l)

    # access compound data structure using indices [ijkm......]
    print(v3[2])
    print(m2[1, 1])


basic_data_structure()

# Domain specific data structures

# Field: this is the most recommended ds we can use in Taichi
# can be read from both taichi and Python scope
# Scalar field,note shape could be 0-d, that makes perfect global variable
gamma = ti.field(dtype=ti.f64, shape=())
gamma[None] = 40  # access by [None], otherwise [i,j,k ...]
velocity_2D = ti.Vector.field(n=2, dtype=ti.f64, shape=())
velocity_2D[None] = ti.Vector([1.0, 1.0])
heat_field = ti.field(dtype=ti.f64, shape=(256, 256))  # Scalar field
# only need to specify 'n' when it is not scalar field
strain_tensor_field = ti.Matrix.field(n=2, m=2, dtype=ti.f32, shape=(64, 64))
gravitational_field = ti.Vector.field(
    n=3, dtype=ti.f32, shape=(512, 512, 512))  # right way for 3D ?
print(gamma)
print(velocity_2D)
print(type(heat_field))
print(type(strain_tensor_field))
print(type(gravitational_field))
