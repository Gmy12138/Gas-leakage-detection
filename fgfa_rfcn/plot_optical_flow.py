from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from imageio import imread, imwrite

def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

fnames1=r'E:\smoke_data\IR_smoke\Data\VID\val\8\000019.png'
fnames2=r'E:\smoke_data\IR_smoke\Data\VID\val\8\000020.png'
taxi1 = cv.imread(fnames1,0)
taxi2 = cv.imread(fnames2,0)

flow = cv.calcOpticalFlowFarneback(taxi1, taxi2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow1 = flow.transpose(2,0,1)
print (flow.shape)


rgb_flow = flow2rgb(20 * flow1, max_value=None)
to_save = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
imwrite('111.png', to_save)

step = 3
plt.quiver(np.arange(0, flow.shape[1], step), np.arange(flow.shape[0], -1, -step),
           flow[::step, ::step, 0], flow[::step, ::step, 1])

plt.show()