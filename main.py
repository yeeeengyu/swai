from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import torch

app = FastAPI()

# Lazily load Whisper model on first request to avoid import-time errors on Windows
def get_whisper_model():
    # cache model on the app state to avoid reloading
    if getattr(app.state, "whisper_model", None) is not None:
        return app.state.whisper_model

    try:
        import whisper
    except Exception as e:
        # Import failed (e.g. whisper tries to load libc on Windows). Raise to be handled by route.
        raise RuntimeError(f"Failed to import whisper: {e}") from e

    try:
        model = whisper.load_model("small")
    except Exception as e:
        raise RuntimeError(f"Failed to load whisper model: {e}") from e

    app.state.whisper_model = model
    return model


@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    try:
        model = get_whisper_model()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language='ko')
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return JSONResponse({"text": result["text"]})

@app.post("/tts")
async def tts(text: str):
    tts = gTTS(text=text, lang='ko')
    tmp_path = tempfile.mktemp(suffix=".mp3")
    tts.save(tmp_path)
    return FileResponse(tmp_path, media_type="audio/mpeg", filename="tts.mp3")
