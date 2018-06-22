FROM ubuntu:latest
#
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

#
RUN pip3 install keras 
RUN pip3 install tensorflow
RUN pip3 install sklearn
RUN pip3 install keras 
RUN pip3 install pandas
RUN pip3 install librosa

COPY data ./data
COPY audio_classifier.py ./audio_classifier.py

ENTRYPOINT ["python3", "audio_classifier.py"]

