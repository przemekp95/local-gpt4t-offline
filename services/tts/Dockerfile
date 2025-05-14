FROM python:3.10-slim

# Systemowe zależności
RUN apt update && apt install -y espeak ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalacja Coqui TTS z XTTS-v2
RUN pip install TTS==0.22.0

# Skopiuj serwer API
COPY server.py .

CMD ["python", "server.py"]

