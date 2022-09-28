from distutils.log import debug
import json
import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi import File, UploadFile, FastAPI
from fastapi.responses import HTMLResponse, Response
import lib.secureqr as secureqr
from lxml import etree as ET
from fastapi.templating import Jinja2Templates
import numpy as np
from PIL import Image
import io
from imageio import v3 as iio


app = FastAPI()
templates = Jinja2Templates(directory="templates")


# @app.get("/generate_sqr", response_class=HTMLResponse)
# async def generateSQR(v:int,box:int,data:str):       
#     return ET.tostring(secureqr.generateSVGSQR(data,v,box))

@app.get("/watermark_sqr", response_class=HTMLResponse)
async def watermarkSQR(v:int,quant:int,data:str):   
    im,wm = secureqr.make_secureQR(data,qr_ver=v,quant=quant)
    img = Image.fromarray(im)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
        
    headers = {'Content-Disposition': 'inline; filename="SQR.png"'}
    return Response(img_byte_arr, headers=headers, media_type='image/png')


@app.post("/verify_wsqr")
def upload(v:int,quant:int,pct:float, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        im = np.asanyarray(Image.open(io.BytesIO(contents)).convert("L"))
        data,s,(p,q) = secureqr.compare2template(im,v,quant,pct)    
        p = p.tolist()
        q = q.tolist()         
        # with open(file.filename, 'wb') as f:
        #     f.write(contents)
    except Exception as e:
        return {"message": "There was an error uploading the file","error":str(e)}
    finally:
        file.file.close()
    

    return {"data":data,"score": s,"actual-dist": p, "scanned-dist": q}
    



if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, debug=True,reload=True)



# from PIL import Image
# import io

# image_data = ... # byte values of the image
# image = Image.open(io.BytesIO(image_data))
# image.show()