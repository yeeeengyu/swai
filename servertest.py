from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os, tempfile
import torch
from TTS.api import TTS

app = FastAPI()

MODEL_PATH  = os.getenv("TTS_MODEL_PATH",  r"C:\Users\pc\Desktop\최인규\건들면사망\swai\vits_pretrained\best_model.pth")
CONFIG_PATH = os.getenv("TTS_CONFIG_PATH", r"C:\Users\pc\Desktop\최인규\건들면사망\swai\vits_pretrained\config.json")

_tts = None

def get_tts():
    global _tts
    if _tts is not None:
        return _tts

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model not found: {MODEL_PATH}")
    if not os.path.exists(CONFIG_PATH):
        raise RuntimeError(f"config not found: {CONFIG_PATH}")

    _tts = TTS(model_path=MODEL_PATH, config_path=CONFIG_PATH, gpu=torch.cuda.is_available())
    return _tts

def cleanup(path: str):
    try: os.remove(path)
    except: pass

@app.on_event("startup")
def warmup():
    # 서버 시작 시 모델 미리 로드 (첫 요청 지연 제거)
    get_tts()

@app.post("/tts")
def tts(text: str, background: BackgroundTasks):
    try:
        tts = get_tts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        tts.tts_to_file(text=text, file_path=out_path)
    except Exception as e:
        cleanup(out_path)
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

    background.add_task(cleanup, out_path)
    return FileResponse(out_path, media_type="audio/wav", filename="tts.wav")
