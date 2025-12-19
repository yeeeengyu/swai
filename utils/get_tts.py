from dotenv import load_dotenv
import os
import torch
from TTS.api import TTS
load_dotenv()

MODEL_PATH  = os.getenv("TTS_MODEL_PATH",  r"C:\Users\User\Desktop\swai_fuck\models\checkpoint.pth")
CONFIG_PATH = os.getenv("TTS_CONFIG_PATH", r"C:\Users\User\Desktop\swai_fuck\models\config.json")

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