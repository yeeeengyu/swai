import streamlit as st
import requests
import tempfile
import sounddevice as sd
import wavio
import io

st.title("청각장애인을 위한 STT/TTS 실시간 대화 지원")
st.header("음성 → 텍스트 ( STT )")

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
        res = requests.post("http://localhost:8000/stt", files={"file": f})

    if "text" not in res.json():
        st.subheader(f"JSONException {res.json()}")
    else:
        text = res.json()["text"]
        st.subheader("인식 결과 : ")
        st.write(text)

        tts_res = requests.post("http://localhost:8000/tts", params=[("text", text.strip())])

        st.write(tts_res.status_code, tts_res.headers.get("content-type"), len(tts_res.content))

        if tts_res.status_code == 422:
            st.error("음성 입력이 되지 않았습니다. 다시 녹음해주세요!", tts_res.text)
        elif tts_res.status_code != 200:
            st.error(tts_res.text)
        else:
            st.audio(tts_res.content, format="audio/wav")

st.header("텍스트 → 음성 ( TTS )")

manual_text = st.text_area("읽어 줄 문장을 입력하세요.", height=120)

if st.button("입력한 문장 읽어주기"):
    if not manual_text.strip():
        st.warning("문장을 입력해주세요.")
    else:
        tts_res = requests.post("http://localhost:8000/tts", params=[("text", manual_text.strip())])
        st.write(tts_res.status_code, tts_res.headers.get("content-type"), len(tts_res.content))

        if tts_res.status_code != 200:
            st.error(f"TTS 요청 실패: {tts_res.text}")
        else:
            audio_data = io.BytesIO(tts_res.content)
            st.audio(audio_data, format="audio/wav")
