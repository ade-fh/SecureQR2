from lib.secureqr import generateSQR, template_matching, percentage_matching
import numpy as np
from PIL import Image
import pandas as pd
import os

box_size = 5
v=3
dir = f"v{v}b{box_size}"
os.makedirs("data/"+dir,exist_ok=True)
datainfo = []
for idx in range(100):
    img = generateSQR(f"halo{idx}",version=v,box_size=box_size)
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
    
    print(f"halo{idx}")
    print(data_string,dot_string)

    match = template_matching(f"halo{idx}",data_string,dot_string,version=v)
    percent = percentage_matching(f"halo{idx}",data_string,dot_string,version=v)
    print(match,percent)

    if match or True:
        Image.fromarray(img*255).save(f"data/{dir}/{idx}.png")
        datainfo.append([f"halo{idx}",data_string,dot_string])

df = pd.DataFrame(datainfo)
df.to_csv(f"data/{dir}.csv",header=False,index=False)
