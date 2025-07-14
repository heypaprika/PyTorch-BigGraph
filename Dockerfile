FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY ./ ./

RUN PBG_INSTALL_CPP=1 pip install -e .

ENTRYPOINT ["python3", "/opt/ml/code/train.py"]
