import taichi as ti
import math
from utils import taichi_utils
import matplotlib.pyplot as plt

# Taichi init
taichi_utils.initialization()

# global vars
order = 6
N = int(math.pow(2,order))
total = N ** 2
mesh_grid = ti.Vector.field(n=2,dtype = ti.f64,shape=(4,))
path = ti.Vector.field(n=2,dtype = ti.f64,shape=(total,))

def hilbert_order(i):
    """one order Hilbert curve

    Args:
        i (_type_): _description_

    Returns:
        _type_: _description_
    """
    mesh_grid[0] = ti.Vector([0,0])
    mesh_grid[1] = ti.Vector([0,1])
    mesh_grid[2] = ti.Vector([1,1])
    mesh_grid[3] = ti.Vector([1,0])
    index = i & 3 # 0011
    v = mesh_grid[index]
    
    for j in range(1,order):    
        i >>= 2
        index = i&3
        size = math.pow(2,j)
        
        if index == 0:
            v[0],v[1] = v[1], v[0]
        elif index == 1:
            v[1] += size
        elif index == 2:
            v[0] += size
            v[1] += size
        elif index == 3:
            temp = size - 1 - v[0]
            v[0] = size - 1 - v[1]
            v[1] = temp
            v[0] += size
    
    return v

for i in range(total):
    path[i] = hilbert_order(i)
    length = 1024 / N
    path[i] *= length
    path[i] += ti.Vector([length/2,length/2])

# plot result
points = path.to_numpy()
x_s = points[:,0]
y_s = points[:,1]

plt.title('Hilbert Curve')
plt.plot(x_s, y_s, linewidth=2)
plt.show()