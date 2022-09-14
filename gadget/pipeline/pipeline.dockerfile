FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:20.09-py3

# Argument defition corresponding to Docker Compose
ARG PACKAGE_VER
ARG PYPI_SERVER

# install dependencies
RUN python3 -m pip install --upgrade pip setuptools

RUN python3 -m pip install gadget_pipeline_server==0.0.0.dev1156055067065066060 --extra-index-url $PYPI_SERVER
# RUN python3 -m pip install gadget_pipeline_server==$PACKAGE_VER --extra-index-url $PYPI_SERVER

COPY ./pipeline_requirements.txt /data/requirements.txt
RUN pip3 install -r /data/requirements.txt


# create the necessary directories
WORKDIR /home/gadget/workspace
RUN mkdir /home/gadget/workspace/image_archive
RUN mkdir /home/gadget/workspace/pipeline


RUN python3 -m pip install --upgrade opencv-python
RUN python3 -m pip install --upgrade opencv-contrib-python
RUN python3 -m pip install --upgrade opencv-python-headless

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
