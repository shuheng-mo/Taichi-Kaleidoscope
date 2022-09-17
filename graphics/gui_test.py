
import taichi as ti
from utils import taichi_utils
import random
import numpy as np

# Taichi initialization
taichi_utils.initialization()
        
gui = ti.GUI('Kaleidoscope')

# items
radius = gui.slider('Radius', 1, 15, step=1)
angle = gui.slider('Angle',0,360,0.01)
xcoor = gui.label('X-coordinate')
ycoor = gui.label('Y-coordinate')
save_btn = gui.button('Save')

colour = 0
xcoor.value = 0.5
ycoor.value = 0.5
radius.value = 10

# # interactive draw
# while gui.running:
#     for e in gui.get_events(gui.PRESS):
#         if e.key == gui.ESCAPE:
#             gui.running = False
#         elif e.key == gui.LMB:
#             x,y = gui.get_cursor_pos()
#             xcoor.value = x
#             ycoor.value = y
#             colour = random.randint(0,1000)
#         elif e.key == save_btn:
#             print('Fractal tree saved')
    
#     gui.line((0.5,0),(0.5,0.5),radius.value,200)
#     # gui.circle((xcoor.value,ycoor.value), radius=radius.value,color=colour)
#     gui.show()