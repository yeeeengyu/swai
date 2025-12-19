from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import os, tempfile
import torch, soundfile as sf
from contextlib import asynccontextmanager
from transformers import VitsModel, AutoTokenizer
from utils.schemas import TTSReq
from utils.clean import cleanup
from utils.getwhisper import get_whisper_model
import whisper

_tts = None
_whisper_model = None

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
tok = None

@asynccontextmanager
async def warmup(app: FastAPI):
    global model, tok
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
    model = VitsModel.from_pretrained("facebook/mms-tts-kor").to(device).eval()
    yield

app = FastAPI(lifespan=warmup)

from fastapi import Query, HTTPException
from fastapi.responses import FileResponse
import os, tempfile, uuid
import numpy as np
import soundfile as sf
import torch
try:
    from uroman import Uroman
    _uroman = Uroman()
    def romanize_ko(s: str) -> str:
        return _uroman.romanize_string(s)
except Exception:
    _uroman = None
    def romanize_ko(s: str) -> str:
        return s
from fastapi import Query, HTTPException
from fastapi.responses import Response
import io
import numpy as np
import soundfile as sf
import torch

@app.post("/tts")
def tts(text: str = Query(...)):
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is empty")

    text = romanize_ko(text)

    inputs = tok(text, return_tensors="pt")
    if inputs["input_ids"].shape[1] == 0:
        raise HTTPException(status_code=422, detail="text produced no tokens")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    audio = out.waveform.detach().cpu().numpy().squeeze()
    if audio.size < 100:
        raise HTTPException(status_code=500, detail="generated audio too short")

    audio = np.asarray(audio, dtype=np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio, 22050, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    try:
        stt_model = get_whisper_model()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = stt_model.transcribe(tmp_path, language="ko")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return JSONResponse({"text": result.get("text", "")})
