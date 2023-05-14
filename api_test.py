# -*- coding: utf-8 -*-
from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from typing import List,Optional
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

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

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

app = FastAPI()
template = Jinja2Templates("templates")
app.mount("/static",StaticFiles(directory="./static"),name="static")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/vits/{txt}")
async def vits(txt: str = ""):
    logger.success(f"VTIS_api服务接收到文本为 【{txt}】 的请求，正在积极回应！")
    s = time.time()

    file_path = Path() / os.getcwd()/ "models" / "lovelive" 
    config = file_path / "config.json"
    model = file_path / "G.pth"
    hps = utils.get_hparams_from_file(config)
    hps_symbols = hps.symbols if 'symbols' in hps.keys() else symbols
    hps.data.text_cleaners = hps.data.text_cleaners if 'text_cleaners' in hps.data.keys() else []
    hps.symbols = hps_symbols
    net_g = SynthesizerTrn(
        len(hps_symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model, net_g)

    file_in_memory = io.BytesIO()
    stn_tst = get_text(txt.replace("\n", ""), hps)
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8,
                            length_scale=1.2)[0][0, 0].data.cpu().float().numpy()
    write(file_in_memory, hps.data.sampling_rate, audio)
    file_in_memory.seek(0)
    encode_output = base64.b64encode(file_in_memory.read())
    gc.collect()
    torch.cuda.empty_cache()
    logger.success(f"VTIS_api服务接收到文本为 【{txt}】处理已完成，耗时{round(time.time() - s, 2)}秒！")
    cretae_py()
    return {"code": 200, "audio": encode_output, "time": (time.time() - s)}

if __name__ == "__main__":
    uvicorn.run(app='api:app',port=25565,reload=True)