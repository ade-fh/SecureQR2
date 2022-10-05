import numpy as np
from .utils.security import passwd, passwd_check
import qrcode
from PIL import Image
import numpy as np
from numpy.random import default_rng
import hashlib
import random
# use lxml for generating the QR manually
from lxml import etree as ET

# SECRET = os.getenv('SECRET')
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
    return img, watermark

import cv2
import numpy as np
from scipy.special import rel_entr
from scipy.spatial import distance

def read_sqr(im):
    qrDecoder = cv2.QRCodeDetector()
    im_filter = np.copy(im)  
    myqr = qrDecoder.detectAndDecode(im_filter)
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


# OLD code
def get_data_ids(n=29):
    ids = np.array([[[j,i] for i in range(n)] for j in range(n)])
    # mask corner
    ids[:8,:8] = n-1
    ids[:8,-8:] = n-1
    ids[-8:,:8] = n-1
    return np.unique(ids.reshape(-1,2),axis=0)[1:]

def get_salt_ids(n=29):
    ids = np.array([[[j,i] for i in range(n)] for j in range(n)])

    # put it in right bottom
    ids[:9,:] = n-1
    ids[:,:-4] = n-1
    return np.unique(ids.reshape(-1,2),axis=0)[1:]

def ids_salt(n=29,seed=0):
    np.random.seed(seed)
    unique_ids = get_salt_ids(n)
    rand = np.random.choice(len(unique_ids),48,replace=False)
    return unique_ids[rand]

  
def get_img_and_dots(data:str, version:int=3):
    border=0
    n = 17 + version*4
    img = qrcode.make(data, version=version,
                  error_correction=qrcode.constants.ERROR_CORRECT_H, 
                  box_size=1,
                  border=0)
    _,salt,hash = passwd(SECRET+str(data)).split(":")   

    np_img = np.asanyarray(img).astype('uint8')

    bsalt = bin(int(salt, 16))[2:]
    bhash = bin(int(hash, 16))[2:]
    bsalt = '0'*(48-len(bsalt)) + bsalt
    bhash = '0'*(160-len(bhash)) + bhash  

    ids = ids_salt(n=n,seed=sum([ord(i) for i in data]))
    for (i,j), b in zip(ids,bsalt):    
        np_img[i,j] = int(b)

    dots = np.copy(np_img)
    # add hash 
    ids = get_data_ids(n)
    mult = len(ids)//len(bhash) + 1
    bhash = (mult*bhash)[:len(ids)]
    for (i,j), b in zip(ids,bhash):
        dots[i,j] = int(b) 
    return np_img, dots

def create_SVG(data:np.ndarray,dots:np.ndarray,box_size):
    offset = 4
    r,c = data.shape
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", version="1.1",
                    height=f"{offset*2+r}mm", width=f"{offset*2+r}mm")

    for i in range(r):
        for j in range(c):
            if not data[i,j]:
                svg.append(ET.Element("rect", height="1mm", width="1mm", 
                            x=f"{offset+j}mm", y=f"{offset+i}mm"))

            if data[i,j]!= dots[i,j]:
                svg.append(ET.Element("circle", r=f"{1/(2*box_size)}mm", 
                            cx=f"{offset+j+0.5}mm", cy=f"{offset+i+0.5}mm",
                            fill='black' if data[i,j] else 'white'))
    return ET.ElementTree(svg)

def generateSVGSQR(data:str, version:int=3, box_size:int=5):
    # ensuring can put salt to QR
    if version < 2:
        return {"error":"version should be 2+"}

    np_img,dots = get_img_and_dots(data,version)
    return create_SVG(np_img,dots,box_size)

# deprecated function use SVG SQR only
def generateSQR(data:str, version:int=3, box_size:int=5):
    border=0
    n = 17 + version*4
    img = qrcode.make(data, version=version,
                  error_correction=qrcode.constants.ERROR_CORRECT_H, 
                  box_size=box_size,
                  border=border)
    
    _,salt,hash = passwd(SECRET+str(data)).split(":")    
    
    
    bsalt = bin(int(salt, 16))[2:]
    bhash = bin(int(hash, 16))[2:]
    bsalt = '0'*(48-len(bsalt)) + bsalt
    bhash = '0'*(160-len(bhash)) + bhash

    np_img = np.asanyarray(img).astype('uint8')

    # add salt key to QR
    ids = ids_salt(n=n,seed=sum([ord(i) for i in data]))
    offset = border*box_size
    for (i,j), b in zip(ids,bsalt):    
        np_img[i*box_size+offset:(i+1)*box_size+offset,
            j*box_size+offset:(j+1)*box_size+offset] = int(b)  
    
    # add hash 
    ids = get_data_ids(n)
    mult = len(ids)//len(bhash) + 1
    bhash = (mult*bhash)[:len(ids)]
    offset = border*box_size + box_size//2
    for (i,j), b in zip(ids,bhash):
        np_img[i*box_size+offset,j*box_size+offset] = int(b)  
    
    return np_img


def get_hash_salt(data,seq,dots, version=3):
    n = 17 + version*4
    seq = np.array([i for i in seq]).reshape(n,n)
    dots = np.array([i for i in dots]).reshape(n,n)

    ids = ids_salt(n=n, seed=sum([ord(i) for i in data]))
    salt = ''
    for i,j in ids:
        salt += seq[i,j]
    
    salt = str(hex(int(salt,2)))[2:]
    salt = '0'*(12-len(salt)) + salt

    ids = get_data_ids(n)[:160]
    hash = ''
    for i,j in ids:
        hash += dots[i,j]
    
    hash = str(hex(int(hash,2)))[2:]
    hash = '0'*(40-len(hash)) + hash
    return hash, salt

def template_matching(data,seq,dots, version=3):
    hash, salt = get_hash_salt(data,seq,dots,version)
    return passwd_check(f'sha1:{salt}:{hash}',SECRET+data)   

def percentage_matching(data,seq,dots, version=3):
    hash, salt = get_hash_salt(data,seq,dots,version)
    hash_diggest = passwd_check(f'sha1:{salt}:{hash}',SECRET+data,exact_match=False) 
    
    num_of_bits=160
    scale=16


    hash = bin(int(hash, scale))[2:].zfill(num_of_bits)
    hash_diggest = bin(int(hash_diggest, scale))[2:].zfill(num_of_bits)

    match = sum([1 if x==y else 0 for x, y in zip(hash, hash_diggest)])/1.6
    return match 

