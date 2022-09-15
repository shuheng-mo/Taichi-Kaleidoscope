
import taichi as ti
from utils import taichi_utils
import random
import numpy as np

# Taichi initialization
taichi_utils.initialization()
        
gui = ti.GUI('Kaleidoscope')

radius = gui.slider('Radius', 1, 50, step=1)
xcoor = gui.label('X-coordinate')
ycoor = gui.label('Y-coordinate')
save_btn = gui.button('Save')

prev_x = xcoor
prev_y = ycoor

colour = 0
xcoor.value = 0.5
ycoor.value = 0.5
radius.value = 10

while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.LMB:
            x,y = gui.get_cursor_pos()
            print("(x,y): ",x,",",y)
            xcoor.value = x
            ycoor.value = y
            colour = random.randint(0,1000)
        elif e.key == 's':
            radius.value -= 1
        elif e.key == 'w':
            radius.value += 1
        elif e.key == save_btn:
            print('Kaleidoscope saved')
    
    gui.circle((xcoor.value,ycoor.value), radius=radius.value,color=colour)
    gui.show()