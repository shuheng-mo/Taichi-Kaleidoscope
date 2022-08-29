from mpi4py import MPI
import taichi as ti
import numpy as np

"""Simple example using Taichi in MPI p2p blocking communication
@author: Shuheng Mo
@time: 2022-08-29 21:10:18
"""

rank = MPI.COMM_WORLD.Get_rank()
num_process = MPI.COMM_WORLD.Get_size()

# here we start taichi for each process
ti.init(ti.cpu)

# Cartesian Topology
comm = MPI.COMM_WORLD.Create_cart(
    dims=(1, 2), periods=(False, False), reorder=True)

# create global field
x = ti.Vector.field(n=1, dtype=ti.f32, shape=(10,))


@ti.kernel
def sum(x: ti.template()):
    for i in ti.grouped(x):
        x[i] += 3.14 * 9 - 27 + 1.94


@ti.kernel
def print_buffer(x: ti.template()):
    for i in ti.grouped(x):
        print(x[i])


if rank == 0:
    # start taichi for specific process
    # ti.init(ti.cpu)
    sum(x)
    send_buffer = x.to_numpy()
    print('Send from 0 to 1:', send_buffer.reshape(1, 10))
    comm.Send(buf=send_buffer, dest=1, tag=11)
    comm.Recv(buf=send_buffer, source=1, tag=22)
    print('Recv from 1', send_buffer.reshape(10,))

else:
    recv_buffer = np.empty((10,), dtype=np.float32)
    comm.Recv(buf=recv_buffer, source=0, tag=11)
    print('Recv from 0', recv_buffer)
    x.from_numpy(recv_buffer.reshape(10, 1))
    sum(x)
    # print_buffer(x)
    recv_buffer = x.to_numpy()
    print('Send from 1 to 0', recv_buffer.reshape(10,))
    comm.Send(buf=recv_buffer, dest=0, tag=22)
