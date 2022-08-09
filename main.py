import json
from fastapi import APIRouter, FastAPI, Request
from fastapi import File, UploadFile, FastAPI
from fastapi.responses import HTMLResponse,FileResponse
import secureqr
from lxml import etree as ET
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/generate_sqr", response_class=HTMLResponse)
async def generateSQR(v:int,box:int,data:str):       
    return ET.tostring(secureqr.generateSVGSQR(data,v,box))

@app.post("/verify_sqr")
async def verify(v:int,data:str,seq:str,dot:str):  
    print("here...")     
    return  {"match_percent": secureqr.percentage_matching(data,seq,dot,version=v)}


# @app.post("/verify_isqr")
# def upload(file: UploadFile = File(...)):
#     try:
#         contents = file.file.read()
#         with open(file.filename, 'wb') as f:
#             f.write(contents)
#     except Exception:
#         return {"message": "There was an error uploading the file"}
#     finally:
#         file.file.close()

#     return {"message": f"Successfully uploaded {file.filename}"}


# from PIL import Image
# import io

# image_data = ... # byte values of the image
# image = Image.open(io.BytesIO(image_data))
# image.show()