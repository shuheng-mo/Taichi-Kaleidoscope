import taichi as ti
from utils import taichi_utils

# init
taichi_utils.initialization()

# sparse data layout (SNode-tree) is used to allocate memory dynamically
x = ti.field(ti.i32)
block1 = ti.root.pointer(ti.i,3)
block2 = block1.dense(ti.j,3)
block2.place(x)

# equals to
# ti.root.pointer(ti.i, 3).dense(ti.j, 3).place(x)

x[1,1] = 1 # suppose we have 1 val located on centre of sparse grid

@ti.kernel
def access_all():
    for i, j in x:
        # x[i, j] = i + j # cannot assign dynamically?
        print("({},{}):".format(i,j),x[i, j])

access_all()

# to check whether a point has been activated, we can use
ti.is_active(block1,[0]) # ? you sure this is correct? it return a expression and logically true for all case

# manually activate and deactivate
# ti.activate()
# ti.deactivate()
# ti.deactivate_all_snodes()