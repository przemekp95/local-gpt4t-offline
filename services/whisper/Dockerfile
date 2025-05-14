FROM debian:bookworm

RUN apt update && apt install -y git build-essential ffmpeg curl cmake python3 python3-pip

WORKDIR /app

# Skopiuj i zbuduj whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git whisper && cd whisper && make

# Skopiuj serwer + model
COPY server.py /app/whisper/
COPY models /app/whisper/models

# Pythonowe zależności
RUN pip3 install flask pydub --break-system-packages

WORKDIR /app/whisper

CMD ["python3", "server.py"]

