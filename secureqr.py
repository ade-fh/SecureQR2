import numpy as np
import utils as iplib
import qrcode
from PIL import Image
import numpy as np

SECRET = "rispro"

def get_data_ids(n=29):
    ids = np.array([[(j,i) for i in range(n)] for j in range(n)])
    # mask corner
    ids[:8,:8] = -1
    ids[:8,-8:] = -1
    ids[-8:,:8] = -1
    return np.unique(ids.reshape(-1,2),axis=0)[1:]

def ids_salt(n=29,seed=0):
    np.random.seed(seed)
    unique_ids = get_data_ids(n)
    rand = np.random.choice(len(unique_ids),48,replace=False)
    return unique_ids[rand]

def generateSQR(data:str, version:int=3, box_size:int=5):
    border=0
    n = 17 + version*4
    img = qrcode.make(data, version=version,
                  error_correction=qrcode.constants.ERROR_CORRECT_H, 
                  box_size=box_size,
                  border=border)
    
    _,salt,hash = iplib.passwd(SECRET+str(data)).split(":")    
    
    print('salt&hash', salt, hash)
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
            j*box_size+offset:(j+1)*box_size+offset] = 0 if b=='0' else 1
    
    # add hash 
    ids = get_data_ids(n)
    mult = len(ids)//len(bhash) + 1
    bhash = (mult*bhash)[:len(ids)]
    offset = border*box_size + box_size//2
    for (i,j), b in zip(ids,bhash):
        bw = np_img[i*box_size+offset,j*box_size+offset]
        np_img[i*box_size+offset,j*box_size+offset] = int(b)  
    
    return np_img


def template_matching(data,seq,dots,box_size, version=3):
    n = 17 + version*4
    seq = np.array([i for i in seq]).reshape(n,n)
    dots = np.array([i for i in dots]).reshape(n,n)

    ids = ids_salt(n=n,seed=sum([ord(i) for i in data]))
    salt = ''
    for i,j in ids:
        salt+= seq[i,j]
    
    salt = str(hex(int(salt,2)))[2:]
    salt = '0'*(12-len(salt)) + salt

    ids = get_data_ids(n)[:160]
    hash = ''
    for i,j in ids:
        hash += dots[i,j]
    hash = str(hex(int(hash,2)))[2:]
    hash = '0'*(40-len(hash)) + hash

    print('salt&hash', salt, hash)

    return iplib.security.passwd_check(f'sha1:{salt}:{hash}',SECRET+data)   
