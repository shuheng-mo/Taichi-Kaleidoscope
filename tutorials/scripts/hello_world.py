import taichi as ti

"""This serves as hello world of Taichi, have a try.
@author:Shuheng Mo
@time:2022-08-27 00:32:58
"""

# Initialize Taichi and run it on CPU (default)
# - `arch=ti.gpu`: Run Taichi on GPU and has Taichi automatically detect the suitable backend
# - `arch=ti.cuda`: For the NVIDIA CUDA backend
# - `arch=ti.metal`: [macOS] For the Apple Metal backend
# - `arch=ti.opengl`: For the OpenGL backend
# - `arch=ti.vulkan`: For the Vulkan backend
# - `arch=ti.dx11`: For the DX11 backend

# ti.init(arch=ti.cpu)
# ti.init(arch=ti.gpu) # falls back to cpu backend automatically if no GPU available
# ti.init(arch=ti.cuda, device_memory_GB=3.4) # allocate exact GPU memory for taichi
# allocate GPU mem by fraction
ti.init(arch=ti.gpu, device_memory_fraction=0.7)

# there are a lot more args for ti.init() we can use

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: float):
    # taichi automatically parallelize the outermost scope range-for,struct-for loop, not the inner ones
    # if there is another statement before this loop, it cannot be parallelized
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        # break # ----------> It is NOT valid to have break in parallelized loops here !
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
            # break # do break here is OK !
        pixels[i, j] = 1 - iterations * 0.02

# range-for: for x in range(999):

# struct-for: for x,y in z:

# Note that for STRUCT-FOR, it CANNOT be in the inner loops, only outermost supported

# code under taichi decorators are in taichi scope, outside it is the Python scope


gui = ti.GUI("Julia Set", res=(n * 2, n))

i = 0
while gui.running:
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
    i = i + 1

# the really diff between ti.kernel and ti.func is:
# ti.kernel can only be called from python
# ti.func can only be called by ti.kernel and other ti.func
# ti.kernel is the smallest unit for runtime execution

# For current ti.kernel arguments:
# 8 args maximum, must type-hinted, scalar values only, pass by value (no ref or pointers)