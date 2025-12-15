from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from utils.getwhisper import get_whisper_model
import tempfile, os, json
import torch
import soundfile as sf
from vits.models import SynthesizerTrn
from vits.text import text_to_sequence

app = FastAPI()
_VITS = {
    "model": None,
    "hps": None,
    "device": None,
}

def _dict_to_ns(d):
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(x) for x in d]
    return d

def get_vits():
    if _VITS["model"] is not None:
        return _VITS["model"], _VITS["hps"], _VITS["device"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG_PATH = "vits/configs/config.json"
    CKPT_PATH   = "vits/checkpoints/G_100000.pth"

    if not os.path.exists(CONFIG_PATH):
        raise RuntimeError(f"config not found: {CONFIG_PATH}")
    if not os.path.exists(CKPT_PATH):
        raise RuntimeError(f"checkpoint not found: {CKPT_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        hps = _dict_to_ns(json.load(f))
    model = SynthesizerTrn(
        hps.data.n_symbols,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=getattr(hps.data, "n_speakers", 0),
        **hps.model.__dict__,
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    _VITS["model"] = model
    _VITS["hps"] = hps
    _VITS["device"] = device
    return model, hps, device


def synthesize_vits(model, hps, device, text: str):
    seq = text_to_sequence(text, getattr(hps.data, "text_cleaners", ["korean_cleaners"]))
    x = torch.LongTensor(seq).unsqueeze(0).to(device)
    x_lens = torch.LongTensor([x.size(1)]).to(device)

    with torch.no_grad():
        y = model.infer(
            x,
            x_lens,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0,
        )[0][0, 0]
    sr = hps.data.sampling_rate
    return y, sr


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
        result = model.transcribe(tmp_path, language="ko")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return JSONResponse({"text": result["text"]})


def _cleanup_file(path: str):
    try:
        os.remove(path)
    except Exception:
        pass


@app.post("/tts")
async def tts(text: str, background: BackgroundTasks):
    try:
        vits_model, hps, device = get_vits()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"VITS load error: {e}")

    try:
        audio, sr = synthesize_vits(vits_model, hps, device, text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VITS synth error: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio.detach().cpu().numpy(), sr)
    background.add_task(_cleanup_file, tmp_path)

    return FileResponse(
        tmp_path,
        media_type="audio/wav",
        filename="tts.wav",
    )
