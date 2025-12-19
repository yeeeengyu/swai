import os

def cleanup(path: str):
    try: os.remove(path)
    except: pass