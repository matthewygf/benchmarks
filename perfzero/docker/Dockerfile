FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 as base

# set up hadoop env
ENV HADOOP_VERSION=2.7.2
LABEL HADOOP_VERSION=2.7.2

# pick up TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-10-0 \
        cuda-cufft-10-0 \
        cuda-curand-10-0 \
        cuda-cusolver-10-0 \
        cuda-cusparse-10-0 \
        libcudnn7=7.4.1.5-1+cuda10.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        libpng-dev \
        pkg-config \
        software-properties-common \
        unzip \
        lsb-core \
        curl \
        wget \ 
        unzip

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda10.0 \
        && apt-get clean

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Add google-cloud-sdk to the source list
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install extras needed by most models
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
        wget \
        htop \
        zip \
        google-cloud-sdk

# Install / update Python and Python3
RUN apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-venv \
        python-yaml \
        python-six \
        python-pip \ 
        python-dev \
        python

# Install ssh and java for openapi
RUN DEBIAN_FRONTEND=noninteractive && \
        apt-get -y update && \
        apt-get -y install automake \
        openjdk-8-jdk \
        openssh-server \
        openssh-client \
        lsof \
        libcupti-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# GET HADOOP
RUN wget -qO- http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz | \
    tar xz -C /usr/local && \
    mv /usr/local/hadoop-${HADOOP_VERSION} /usr/local/hadoop

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
    HADOOP_INSTALL=/usr/local/hadoop \
    NVIDIA_VISIBLE_DEVICES=all

# set HADOOP PATHS
ENV HADOOP_PREFIX=${HADOOP_INSTALL} \
    HADOOP_BIN_DIR=${HADOOP_INSTALL}/bin \
    HADOOP_SBIN_DIR=${HADOOP_INSTALL}/sbin \
    HADOOP_HDFS_HOME=${HADOOP_INSTALL} \
    HADOOP_COMMON_LIB_NATIVE_DIR=${HADOOP_INSTALL}/lib/native \
    HADOOP_OPTS="-Djava.library.path=${HADOOP_INSTALL}/lib/native"

# set CUDA and NVIDIA PATHs
ENV CUDA_BIN_PATH=/usr/local/cuda/bin \
    NVIDIA_BIN_PATH=/usr/local/nvidia/bin \
    NVIDIA_LIB_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    CUDA_LIB_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs

ENV PATH=${NVIDIA_BIN_PATH}:${CUDA_BIN_PATH}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${HADOOP_BIN_DIR}:${HADOOP_SBIN_DIR} \
    LD_LIBRARY_PATH=${NVIDIA_LIB_PATH}:${CUDA_LIB_PATH}:${JAVA_HOME}/jre/lib/amd64/server:${LD_LIBRARY_PATH}


# Setup Python3 environment
RUN pip3 install --upgrade pip==9.0.1
# setuptools upgraded to fix install requirements from model garden.
RUN pip3 install --upgrade setuptools google-api-python-client pyyaml google-cloud google-cloud-bigquery
RUN pip3 install wheel absl-py
# NOTE: tf-nightly-gpu is version 1.xx 
RUN pip3 install --upgrade --force-reinstall tf-nightly-gpu

RUN curl https://raw.githubusercontent.com/tensorflow/models/master/official/requirements.txt > /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Add symbol link for libcuda.so
RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
          /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1

WORKDIR /root