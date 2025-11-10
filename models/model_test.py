import whisper
import logging
logger = logging.getLogger(__name__)
model = whisper.load_model("medium")
logger.info("남자 거리차이 테스트사운드")
result = model.transcribe("C:\\Users\\User\\Desktop\\SWAI\\거리차이_남자.wav")
print(result['text'])