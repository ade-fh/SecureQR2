import numpy as np
import matplotlib.pyplot as plt
import qrcode
from PIL import Image
import hashlib
import random
import pathlib
from pathlib import Path
import cv2

SECRET = 'rispro'
# SALT = ''.join(format(ord(x), 'x') for x in SECRET)

def generate_watermark(data,secret,quant=16,size=(100,100),rprop=100):
    dist  = [int((i+0.5)*256/quant) for i in range(quant)]    
    s = data+secret    
    seed = abs(int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16))%2**32
    random.seed(seed) 
    p=random.choices([i for i in range(rprop)], k = quant)
    watermark = random.choices(dist,weights=p,k=size[0]**2)
    watermark = np.array(watermark).reshape(size)
    return watermark.astype('uint8')

def make_secureQR(data:str,secret:str=SECRET,                  
                  pct_wm_size:int=0.3, 
                  bx_size:int=10,
                  border:int=4,
                  qr_ver:int=None,
                  **kwargs)-> np.ndarray:

    img = qrcode.make(data,version=qr_ver, box_size=bx_size, border=border, error_correction=qrcode.constants.ERROR_CORRECT_H)
    img = np.asanyarray(img).astype('uint8')*255
    
    wm_size = (img.shape[0]//bx_size - 2*border) * pct_wm_size
    wm_size = int(wm_size)
    if not wm_size%2==0: wm_size-=1 

    wm_size *= bx_size
    size = (wm_size,wm_size)
    watermark = generate_watermark(data,secret,size=size,**kwargs)
    wm = np.pad(watermark, bx_size//2, constant_values=255)

    # add border watermark 8
    length = wm.shape[0] 
    loc = img.shape[0]//2 - length//2
    img[loc:loc+length,loc:loc+length]= wm
    img[0] = 0
    img[369] = 0
    for i in range(370):
        img[i][0] = 0
        img[i][369] = 0
    return img, watermark

import cv2
import numpy as np
from scipy.special import rel_entr
from scipy.spatial import distance

def read_sqr(im):
    qrDecoder = cv2.QRCodeDetector()
    
    myqr = qrDecoder.detectAndDecode(im)
    if myqr[0]=="": 
        im_filter = np.copy(im) 
        myqr = qrDecoder.detectAndDecode(cv2.medianBlur(im_filter,ksize=3))
    
    top = myqr[1][0][0].astype(int)
    bottom = myqr[1][0][2].astype(int)+1
    qim = im[top[0]:bottom[0],top[1]:bottom[1]].astype(float)

    # # normalize
    qim -= qim.min()
    qim *= 255/qim.max()

    return myqr[0],qim

def JS_div(p,q):
    m = (p+q)/2
    div = 0.5*rel_entr(p,m) + 0.5*rel_entr(q,m)
    return div.sum()

def compute_KL(qim,scan,quant=16,pct=0.2,metric='JS'):
    pq = []
    for q in [qim,scan]:
        l = q.shape[0]
        half = l//2
        r = int(0.5*pct*l)             
        wtm = q[half-r:half+r,half-r:half+r]
        wtm = cv2.Sobel(wtm,cv2.CV_64F,1,1,ksize=3)
        dist, *note = np.histogram(wtm.ravel(),bins=quant)
        p = dist/dist.sum()
        pq.append(p)        
    return rel_entr(*pq).sum() if metric=="KL" else distance.jensenshannon(*pq), pq  

def compare2template(im, qr_ver = 3, quant=16, pct=0.2, metric='JS'):
    data ,qim_scan = read_sqr(im) 
    img,_ = make_secureQR(data, qr_ver = qr_ver, quant=quant)
    _,qim = read_sqr(img) 
    return (data,)+compute_KL(qim,qim_scan,quant,pct,metric)