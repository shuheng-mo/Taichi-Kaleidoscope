'''
 # @ Author: Shuheng Mo
 # @ Create Time: 2022-09-25 21:54:40
 # @ Modified by: Shuheng Mo
 # @ Modified time: 2022-09-25 22:03:42
 # @ Description:
 '''

# Advanced topics
# https://docs.taichi-lang.org/docs/quant#quantized-fixed-point-numbers

import taichi as ti
from utils import taichi_utils

# init
taichi_utils.initialization()

# the reason we mention this here is that digits represented by binary numbers in computer
# to avoid accuracy issue, we tackle repeated decimals/binary numbers
# inexact numbers cause roud-off errors to result not meaningfully

# real max = 1.7977e38 and real min 2.2225e-38
# for data larger than this, you will have to think
# Therefore we consider quantized fixed point of number (though this looks like trash in Taichi for now ... )

i5 = ti.types.quant.int(bits=5)  # 10 bit signed integer
f15 = ti.types.quant.float(exp=5, frac=10, signed=True,
                           compute=ti.f32)  # float not supported yet,frac indicates the significant fraction
fixed_1 = ti.types.quant.fixed(
    10, signed=True, max_value=1.0, compute=None, scale=None)

N = 128
M = 32
x = ti.field(dtype=fixed_1)
y = ti.field(dtype=fixed_1)
ti.root.dense(ti.i, N // M).quant_array(ti.i, M, max_num_bits=M).place(x)
ti.root.dense(ti.i, N // M).quant_array(ti.i, M, max_num_bits=M).place(y)


@ti.kernel
def assign_vectorized():
    # ti.loop_config(bit_vectorize=True)
    for I in ti.grouped(x):
        x[I] = ti.random(ti.i32)
        y[I] = x[I]
        print(I, y[I])


assign_vectorized()
