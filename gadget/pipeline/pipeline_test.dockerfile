FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:20.09-py3
WORKDIR /home/gadget/workspace
COPY ./pipeline/pipeline_requirements.txt /home/gadget/workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

COPY ./data/test_images /home/gadget/workspace/data/test_images