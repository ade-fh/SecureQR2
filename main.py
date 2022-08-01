import json
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse,FileResponse
import secureqr
from lxml import etree as ET
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/generate_sqr", response_class=HTMLResponse)
async def generateSQR(v:int,box:int,data:str):       
    return ET.tostring(secureqr.generateSVGSQR(data,v,box))

@app.get("/verify_sqr", response_class=HTMLResponse)
async def verify(v:int,data:str,seq:str,dot:str):       
    return {'status':'fake'}





