import taichi as ti
from utils import taichi_utils
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


"""
author: Shuheng Mo
time: 2022-09-17 13:41:43
"""

# Taichi init
taichi_utils.initialization()

width = int(input('Enter the width: '))
height = int(input('Enter the height: '))

# a 2D array is needed
sand_field = ti.field(dtype=ti.f64,shape=(width,height)) # a scalar field
next_field = ti.field(dtype=ti.f64,shape=(width,height))
pixel_field = ti.field(dtype=ti.f64,shape=(width,height))


@ti.kernel
def initial_condition():
    sand_field[width//2, height//2] = 200000
    
@ti.func
def copy_2d(x: ti.template(), y: ti.template()):
    for i, j in x:
        y[i, j] = x[i, j]
                
@ti.kernel
def render():
    for x,y in sand_field:
        sand = sand_field[x,y]
        if sand == 0:
            pixel_field[x,y] = 0
        elif sand == 1:
            pixel_field[x,y] = 10
        elif sand == 2:
            pixel_field[x,y] = 20
        elif sand == 3:
            pixel_field[x,y] = 30

@ti.kernel
def update():
    for x,y in sand_field:
        sand = sand_field[x,y]
        if sand < 4:
            next_field[x,y] = sand_field[x,y]  
              
    for x,y in sand_field:
        sand = sand_field[x,y]
        if sand >= 4:
            ti.atomic_add(next_field[x,y],(sand-4))
            if x + 1 < width:
                next_field[x-1,y]+=1
            if x-1 >=0:
                next_field[x+1,y]+=1
            if y+1 < height:
                next_field[x,y-1]+=1
            if y-1 >=0:
                next_field[x,y+1]+=1
    
    copy_2d(next_field,sand_field) # update the field

def time_stepping(t):
    """_summary_

    Args:
        t (_type_): _description_
    """
    for i in range(t):
        render()
        for _ in range(10):
            update()    
        print('[TIME STEP] {}'.format(i))
        result = pixel_field.to_numpy()
        plt.imshow(result,cmap='tab20')
        plt.savefig('output/sand_piles_{}.png'.format(str(i).zfill(3)))

if __name__ == '__main__':
    initial_condition()
    print('START RUNNING ...')
    start_time = perf_counter()
    time_stepping(1000)
    end_time = perf_counter()
    print('END RUNNING ...')
    print('[TOTAL TIME] {}'.format(end_time - start_time))