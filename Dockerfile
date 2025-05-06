FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /tfg

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser" ]
