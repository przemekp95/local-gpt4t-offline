from flask import Flask, request, send_file
import os
from torch.serialization import add_safe_globals

# Dodajemy wymagane klasy do globalnego kontekstu dla torch.load
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

from TTS.api import TTS

app = Flask(__name__)

# Ładowanie modelu XTTS-v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text")
    lang = data.get("lang", "pl")
    speaker_path = "/app/speaker.wav" if os.path.exists("/app/speaker.wav") else None

    output_path = "output.wav"

    tts.tts_to_file(
        text=text,
        file_path=output_path,
        language=lang,
        speaker_wav=speaker_path
    )

    return send_file(output_path, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

