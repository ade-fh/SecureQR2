from secureqr import generateSQR, template_matching
import numpy as np

box_size = 5
v=2
img = generateSQR("halo",version=v,box_size=box_size)
half = box_size // 2 
image = np.asanyarray(img) 

data_string = ''
dot_string = ''
for i in range(0,image.shape[0],box_size):
    for j in range(0,image.shape[1],box_size):
        if image[i,j]>0.5: data_string+='1'
        else: data_string+='0'
        if image[i+half,j+half]>0: dot_string+='1'
        else: dot_string+='0'

match = template_matching("halo",data_string,dot_string,box_size,version=v)
print(match)