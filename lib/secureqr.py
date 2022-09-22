import numpy as np
from .utils.security import passwd, passwd_check
import qrcode
from PIL import Image
import numpy as np

# use lxml for generating the QR manually
from lxml import etree as ET

# SECRET = os.getenv('SECRET')
SECRET = 'rispro'
# SALT = ''.join(format(ord(x), 'x') for x in SECRET)

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

