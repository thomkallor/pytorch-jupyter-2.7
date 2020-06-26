FROM ubuntu:14.04
ENV DEBIAN_FRONTEND=noninteractive

ARG THEANO_VERSION=rel-0.8.2
ARG TENSORFLOW_VERSION=0.6.0
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=1.2.0
ARG LASAGNE_VERSION=v0.1
ARG TORCH_VERSION=latest
ARG CAFFE_VERSION=master
ARG PYTORCH_VERSION=0.3.0

## install basic dependencies
RUN apt-get update && apt-get install -y \
	curl \
    git \
    g++ \
    libssl-dev \
    software-properties-common \
    sudo \
    wget

## install nvidia-375-driver
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN apt-get update && apt-get install -y nvidia-375

## install cuda8.0
RUN wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
RUN yes | dpkg -i ./cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
RUN apt-get update && apt-get install -y cuda-toolkit-8-0
RUN rm cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb

ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"
ENV PATH="/usr/local/cuda/bin:$PATH"

#CuDNN installation cudnn6
RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6_6.0.21-1+cuda8.0_amd64.deb && \
	wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
RUN yes | dpkg -i ./libcudnn6_6.0.21-1+cuda8.0_amd64.deb && \
	yes | dpkg -i ./libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
RUN apt-get update && apt-get install -y libcudnn6-dev
RUN rm libcudnn6_6.0.21-1+cuda8.0_amd64.deb && \
	rm libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb


# conda required for pytorch installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh \
    && bash Miniconda2-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda2-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda2/bin:${PATH}"

## install pip
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python-dev \
	python-tk \
    python-pip

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python-numpy \
		python-scipy \
		python-nose \
		python-h5py \
		python-skimage \
		python-matplotlib \
		python-pandas \
		python-sklearn \
		python-sympy \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

## Dependencies for torch
RUN pip install --user --upgrade \
		ipykernel \
		tornado==4.5.3 \
		six==1.15.0 \
		pyparsing==2.4.7

# Add SNI support to Python
#install useful packages
RUN pip --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1 \
		Cython \
		path.py \
		Pillow \
		jupyter \
		pygments \
		sphinx \
        wheel

# pytorch dependencies 
RUN conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi

# install pytorch
RUN git clone https://github.com/pytorch/pytorch.git
RUN cd pytorch && git checkout "v$PYTORCH_VERSION"
RUN cd pytorch && git submodule sync && git submodule update --init --recursive
RUN export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
RUN cd pytorch && python setup.py install

# Install Torch
RUN git clone --recursive https://github.com/torch/distro.git /torch && \
	cd torch && \
	bash install-deps && \
	yes | bash ./install.sh

# Install Theano and set up Theano config (.theanorc) for CUDA and OpenBLAS
RUN pip --no-cache-dir install git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
	\
	echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN \
		\n[lib]\ncnmem=0.95 \
		\n[nvcc]\nfastmath=True \
		\n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
		\n[DebugMode]\ncheck_finite=1" \
	> /root/.theanorc

## install pylearn2
RUN git clone git://github.com/lisa-lab/pylearn2.git && \
     cd pylearn2 && \
     python setup.py develop
ENV PYLEARN2_DATA_PATH=/data/lisa/data

## install lasagne
RUN pip --no-cache-dir install git+git://github.com/Lasagne/Lasagne.git@${LASAGNE_VERSION}

## caffe needs gcc5
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y gcc-5 g++-5

## Dependencies of caffe
RUN apt-get update && apt-get install -y \
		libboost-all-dev \
		libgflags-dev \
		libgoogle-glog-dev \
		libhdf5-serial-dev \
		libleveldb-dev \
		liblmdb-dev \
		libopencv-dev \
		libprotobuf-dev \
		libsnappy-dev \
		protobuf-compiler \
		libopenblas-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install Caffe
RUN git clone -b ${CAFFE_VERSION} --depth 1 https://github.com/BVLC/caffe.git /root/caffe && \
	cd /root/caffe && \
	cat python/requirements.txt | xargs -n1 pip install && \
	mkdir build && cd build && \
	cmake -DUSE_CUDNN=1 -DBLAS=Open .. && \
	make -j"$(nproc)" all && \
	make install

# Set up Caffe environment variables
ENV CAFFE_ROOT=/root/caffe
ENV PYCAFFE_ROOT=$CAFFE_ROOT/python
ENV PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH \
	PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig


## install cuda 7.0 and cudnn6.5 for tensorflow
RUN wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1410-7-0-local_7.0-28_amd64.deb
RUN yes | dpkg -i ./cuda-repo-ubuntu1410-7-0-local_7.0-28_amd64.deb
RUN apt-get update && apt-get install -y cuda-toolkit-7-0
RUN rm cuda-repo-ubuntu1410-7-0-local_7.0-28_amd64.deb
ENV LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH"

RUN git clone https://github.com/thomkallor/cudnn-6.5.git
ENV LD_LIBRARY_PATH="/root/cudnn-6.5:$LD_LIBRARY_PATH"

# install tensorflow
RUN pip install --no-cache-dir install \
	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

# Set up notebook config
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
COPY run_jupyter.sh /root/

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

WORKDIR "/root"
CMD ["/bin/bash"]