FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Change root user to use 'apt-get'
# USER root 
# RUN sudo apt-get update && \
# apt-get install -y libpq-dev libmysqlclient-dev gcc build-essential

# pip install 
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
RUN pip install -r requirements.txt
WORKDIR /workspace/fasion_graph