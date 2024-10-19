# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
# RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
# RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y git libopencv-dev
# RUN pip install pyyaml Pillow numpy opencv-contrib-python scikit-image
# RUN pip install segmentation-models-pytorch
# RUN pip install pandas tensorboard

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3 python3-distutils curl git python-is-python3 python3-dev git libopencv-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip cache purge
RUN pip install pyyaml Pillow numpy opencv-contrib-python scikit-image segmentation-models-pytorch pandas tensorboard
RUN pip install scipy
