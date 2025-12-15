from fastapi import FastAPI; app = FastAPI()

def get_whisper_model():
    if getattr(app.state, "whisper_model", None) is not None:
        return app.state.whisper_model

    try:
        import whisper
    except Exception as e:
        raise RuntimeError(f"Failed to import whisper: {e}") from e

    try:
        model = whisper.load_model("small")
    except Exception as e:
        raise RuntimeError(f"Failed to load whisper model: {e}") from e

    app.state.whisper_model = model
    return model