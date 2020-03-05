FROM nvidia/cuda:10.0-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo && \
  rm -rf /var/lib/apt/lists/*

# Install python
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# Install java
RUN sudo apt-get update && \
    sudo apt-get install -y openjdk-8-jdk && \
    sudo apt-get install -y ant && \
    sudo apt-get clean;

# Fix certificate issues
RUN sudo apt-get update && \
     sudo apt-get install ca-certificates-java && \
     sudo apt-get clean && \
     sudo update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tensorboard cython
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV FORCE_CUDA="1"
RUN pip install git+https://github.com/facebookresearch/detectron2
# install biodetectron
RUN pip install git+https://github.com/CALM-LMU/BioDetectron.git

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
