import streamlit as st
import requests
import tempfile
import sounddevice as sd
import wavio
import io
st.title("청각장애인을 위한 STT/TTS 실시간 대화 지원")

duration = st.slider("녹음 시간(초)", 1, 10, 3)
if st.button("말하기 시작"):
    st.info("녹음 중입니다...")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    st.success("녹음 완료")

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(tmpfile.name, audio, fs, sampwidth=2)

    with open(tmpfile.name, "rb") as f:
        res = requests.post(f"http://localhost:8000/stt", files={"file": f})
    print(res)
    if "text" not in res.json():
        st.subheader(f"JSONException {res.json()}")
    else: text = res.json()["text"]
    st.subheader("결과 : ")
    st.write(text)

    tts_res = requests.post(f"http://localhost:8000/tts", params={"text": text})
    audio_data = io.BytesIO(tts_res.content)
    st.audio(audio_data, format="audio/mp3")