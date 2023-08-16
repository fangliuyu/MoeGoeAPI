# -*- coding: utf-8 -*-
from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from typing import Optional
import base64
import gc
import io
import os
import uvicorn
from scipy.io.wavfile import write
import torch
from pathlib import Path
import commons
import utils
import re
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from text.symbols import symbols
import time
from loguru import logger

import sys

import argparse
parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--host','-g',type=str, default='127.0.0.1',help='host of the api')
parser.add_argument('--port','-p',type=int, default=8000,help='port of the api')
parser.add_argument('--static','-s',type=str, default='./static',help='File Path of the WAV File Saving')
parser.add_argument('--template','-t',type=str, default='templates',help='File Path of the HTML File Loading')
args = parser.parse_args()
api_host = args.host
api_port = args.port
static_file_path = args.static
template_file_path = args.template

file_path = Path() / os.getcwd()
 
models = {}
model_path = file_path/Path("models")
for file_name in os.listdir(file_path/Path("models")):
    if os.path.isdir(model_path/file_name):
        hps = utils.get_hparams_from_file("./models"/Path(file_name)/"config.json")
        speakers = hps.speakers if 'speakers' in hps.keys() else ['single entity']
        models.update({file_name:speakers})
if len(models) == 0:
    print("没有任何模型,请在models文件夹里放入模型")
    sys.exit() 
def_model = next(iter(models))
def_speakers = models.get(next(iter(models)))[0]
 

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            logger.error(f'Invalid {warning_name}!')
            value = default
    else:
        value = default
    return value, text

def ex_print(text, escape=False):
    if escape:
        return text.encode('unicode_escape').decode()
    else:
        return text

a = 0
def cretae_py():
    global a
    a = a + 1
    if a <= 10:
        pass
    else:
        filePrefix = 'Test'
        fileSuffix = '.py'
        filename = filePrefix + fileSuffix
        with open(filename, 'w') as f:
            f.write('')
        a = 0

def api_for_main(req :Request,models: str = def_model,speaker_id:int=0,text: str = "",escape:bool=False,type: str = "",raw_text: str = "",output:str=""):
    s = time.time()
    # 载入模型信息
    if models == "":
        return {"code":-1, "message": 'models is a void value!' }
    model = file_path/"models"/Path(models)/"G.pth"
    config = file_path/"models"/Path(models)/"config.json"
    # 用户数据
    try:
        speaker_id = int(speaker_id)
    except:
        return  {"code":-1, "message": str(speaker_id) + ' is not a valid ID!' }
    
    if text == "":
        return {"code":-1, "message": 'text is a void value!' }
    
    if text == '[ADVANCED]' and raw_text == "":
        return {"code":-1, "message": 'raw_text is a void value!' }
    
    if type == "wav" and output == "":
        return {"code":-1, "message": 'output is a void value!' }

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    hps_ms.symbols = hps_ms.symbols if 'symbols' in hps_ms.keys() else symbols
    n_symbols = len(hps_ms.symbols)
    hps_ms.data.text_cleaners = hps_ms.data.text_cleaners if 'text_cleaners' in hps_ms.data.keys() else []
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['single entity']
    if speaker_id >= len(speakers):
        return {"code":-1, "message": str(speaker_id) + ' is over speakers number!' }
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)
    try:
        if n_symbols != 0:
            while True:
                message = "successfully"
                if text == '[ADVANCED]':
                    text = raw_text
                    message='Cleaned text is:'+ex_print(_clean_text(text, hps_ms.data.text_cleaners), escape)
                    continue
                length_scale, text = get_label_value(text, 'LENGTH', 1.2, 'length scale')
                noise_scale, text = get_label_value(text, 'NOISE', 0.667, 'noise scale')
                noise_scale_w, text = get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
                stn_tst = get_text(text, hps_ms)
                with torch.no_grad():
                    x_tst = stn_tst.unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
                    sid = torch.LongTensor([speaker_id])
                    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                           noise_scale_w=noise_scale_w, 
                                           length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
                
                if type == "json":
                    file_in_memory = io.BytesIO()
                    write(file_in_memory, hps_ms.data.sampling_rate, audio)
                    file_in_memory.seek(0)
                    encode_output = base64.b64encode(file_in_memory.read())
                    gc.collect()
                    torch.cuda.empty_cache()
                    cretae_py()
                    return {"code": 200,"ID":speaker_id, "speakers":ex_print(speakers[speaker_id],escape), "time": (time.time() - s),"message":message, "audio": encode_output}
                else:
                    file_name = "static/audios"/Path(output).with_suffix(".wav")
                    write(file_name, hps_ms.data.sampling_rate, audio)
                    return FileResponse(file_name)
                
    except Exception as e:
        return {"code":400,"message":str(e)}




app = FastAPI()
template = Jinja2Templates(template_file_path)
app.mount("/static",StaticFiles(directory=static_file_path),name="static")

@app.get("/")
def read_root(req :Request):
    return template.TemplateResponse("index.html",context={"request":req,"models":models,"enumerate":enumerate})

class Item(BaseModel):
    model:Optional[str] = def_model
    speaker:Optional[str] = def_speakers
    url_pic:Optional[bool] = False
    switch:Optional[str] = ""
    text:Optional[str] = ""
    language:Optional[str] = "中日混合"
    length_scale:Optional[float] = 1.2
    noise_scale:Optional[float] = 0.667
    noise_scale_w:Optional[float] = 0.8
    wavdata:Optional[str] = ""

@app.get("/dev")
async def devIndex(req :Request):
    models = {}
    for file_name in os.listdir(file_path/Path("models")):
        if os.path.isdir(model_path/file_name):
            hps = utils.get_hparams_from_file("./models"/Path(file_name)/"config.json")
            speakers = hps.speakers if 'speakers' in hps.keys() else ['single entity']
            models.update({file_name:speakers})
    def_model = next(iter(models))
    def_speakers = models.get(next(iter(models)))[0]
    item = Item(model=def_model,speaker=def_speakers)
    return template.TemplateResponse("dev.html",context={
        "request":req,
        "models":models,
        "item":item,
    })

@app.post("/dev/main")
async def dev(req :Request,
        model:Optional[str] = Form(...),
        speaker:Optional[str] = Form(...),
        switch:Optional[str] = Form(...),
        text:Optional[str] = Form(""),
        language:Optional[str] = Form(...),
        length_scale:Optional[float] = Form(...),
        noise_scale:Optional[float] = Form(...),
        noise_scale_w:Optional[float] = Form(...),
        wavdata:Optional[str] = Form(""),
    ):
    item = Item(model=model,speaker=speaker,switch=switch,text=text,language=language,
                length_scale=length_scale,noise_scale=noise_scale,noise_scale_w=noise_scale_w,wavdata=wavdata)
    if item.url_pic == False and os.path.exists("./static/img"/Path(model)/Path(speaker+".png")):
            item.url_pic = True
    if item.switch == "Submit" :
        return template.TemplateResponse("dev.html",context={
            "request":req,
            "models":models,
            "item":item,
        })
    if item.text == "" or len(item.text)>100:
        item.text='请输入100字符内的文本'
        return template.TemplateResponse("dev.html",context={
            "request":req,
            "models":models,
            "item":item,
        })
    logger.info(f"接收到【{req.client.host}】 的请求:【{item.model,item.speaker,item.text}】，正在积极回应！")
    if item.language == "中文":
        speak_text = "[ZH]"+item.text+"[ZH]"
    elif item.language == "日语":
        speak_text = "[JP]"+item.text+"[JP]"
    else:
        speak_text = item.text

    gpth = file_path/"models"/item.model/"G.pth"
    config = file_path/"models"/item.model/"config.json"
    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    hps_ms.symbols = hps_ms.symbols if 'symbols' in hps_ms.keys() else symbols
    n_symbols = len(hps_ms.symbols)
    hps_ms.data.text_cleaners = hps_ms.data.text_cleaners if 'text_cleaners' in hps_ms.data.keys() else []

    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False
    speaker_id = 0
    for spkeaker_name in models[item.model]:
        if spkeaker_name == item.speaker:
            break
        speaker_id +=1
    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(gpth, net_g_ms)
    if n_symbols != 0:
        scale_length, speak_text = get_label_value(speak_text, 'LENGTH', float(item.length_scale), 'length scale')
        scale_noise, speak_text = get_label_value(speak_text, 'NOISE', float(item.length_scale), 'noise scale')
        scale_noise_w, speak_text = get_label_value(speak_text, 'NOISEW', float(item.noise_scale_w), 'deviation of noise')
        stn_tst = get_text(speak_text, hps_ms)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            sid = torch.LongTensor([speaker_id])
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=scale_noise,
                                    noise_scale_w=scale_noise_w, 
                                    length_scale=scale_length)[0][0, 0].data.cpu().float().numpy()
                
            file_in_memory = io.BytesIO()
            write(file_in_memory, hps_ms.data.sampling_rate, audio)
            file_in_memory.seek(0)
            encode_output = base64.b64encode(file_in_memory.read())
            gc.collect()
            torch.cuda.empty_cache()
            cretae_py()
            item.wavdata = bytes.decode(encode_output)
            return template.TemplateResponse("dev.html",context={
                "request":req,
                "models":models,
                "item":item,
            })
    return template.TemplateResponse("dev.html",context={
        "request":req,
        "models":models,
        "item":item,
    })

@app.get("/{models}")
async def moegoe_for_N(req :Request,models: str="",text: str="",escape:bool=False,type: str="wav",raw_text:str="",output:str=""):
    return api_for_main(req,models,0,text,escape,type,raw_text,output)

@app.get("/{models}/{id}")
async def moegoe_for_N(req :Request,models: str="",id:int=0,text: str="",escape:bool=False,type: str="wav",raw_text:str="",output:str=""):
    return api_for_main(req,models,id,text,escape,type,raw_text,output)

if __name__ == "__main__":
    uvicorn.run(app='api:app',host=api_host,port=api_port,reload=True)