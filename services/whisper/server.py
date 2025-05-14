from flask import Flask, request, jsonify
import subprocess
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return "No file provided", 400

    audio = request.files["file"]
    filename = os.path.join(UPLOAD_FOLDER, datetime.now().isoformat().replace(":", "-") + "_" + audio.filename)
    audio.save(filename)

    result = subprocess.run(
        ["./build/bin/whisper-cli", "-m", "models/ggml-base.en.bin", "-f", filename, "-l", "en", "-nt"],
        capture_output=True,
        text=True
    )

    return jsonify({"transcript": result.stdout.strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

