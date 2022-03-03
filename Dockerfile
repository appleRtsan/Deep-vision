FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN mkdir /build && mkdir dv && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y htop 

COPY ["requirements.txt", "/build/"]
RUN pip install -r /build/requirements.txt
