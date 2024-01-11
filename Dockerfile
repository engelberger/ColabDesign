FROM nvidia/cuda:11.1.1-runtime-ubuntu18.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    build-essential

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:${PATH}

# Create conda environment and install dependencies
#COPY env/SE3nv.yml /tmp/SE3nv.yml



RUN git clone https://github.com/RosettaCommons/RFdiffusion.git
RUN conda env create -f /RFdiffusion/env/SE3nv.yml 
# Activate conda environment
RUN echo "source activate SE3nv" > ~/.bashrc
ENV PATH /opt/conda/envs/SE3nv/bin:$PATH

RUN cd RFdiffusion/env/SE3Transformer && pip install --no-cache-dir -r requirements.txt && python setup.py install

RUN cd RFdiffusion && pip install -e .

RUN apt-get update && apt-get install aria2 -y