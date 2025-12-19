from pydantic import BaseModel

class TTSReq(BaseModel):
    text: str
    output: str = 'out.wav'