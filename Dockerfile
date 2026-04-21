# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 AS base
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=noetic

# preseed tzdata, update package index, upgrade packages and install needed software
RUN truncate -s0 /tmp/preseed.cfg; \
    echo "tzdata tzdata/Areas select Europe" >> /tmp/preseed.cfg; \
    echo "tzdata tzdata/Zones/Europe select Berlin" >> /tmp/preseed.cfg; \
    debconf-set-selections /tmp/preseed.cfg && \
    rm -f /etc/timezone /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata
RUN apt-get update && apt-get install -y \
    unzip \
    nano \
    build-essential \
    git \
    byobu \
    curl \
    wget \
    xclip \
    python3 \
    python3-pip \
    make \
    cmake \
    libeigen3-dev

# Additional dependencies for CMRNext are now handled via requirements.txt
RUN pip3 install --no-cache-dir \
    numpy \
    scikit-image \
    pyquaternion \
    tqdm \
    python-dateutil==2.8.2 \
    open3d \
    pillow

WORKDIR /root/opencv
RUN wget https://github.com/opencv/opencv/archive/4.7.0.zip
RUN unzip 4.7.0.zip
RUN git clone https://github.com/opencv/opencv_contrib.git
WORKDIR /root/opencv/opencv_contrib
RUN git checkout 4.7.0
# Remove all unnecessary modules
RUN rm -rf ./modules/alphamat/ ./modules/aruco/ ./modules/barcode/ ./modules/bgsegm/ ./modules/bioinspired/ ./modules/ccalib/ ./modules/cnn_3dobj/ ./modules/cudabgsegm/ ./modules/cudacodec/ ./modules/cudafeatures2d/ ./modules/cudaobjdetect/ ./modules/cudastereo/ ./modules/cvv/ ./modules/datasets/ ./modules/dnn_objdetect/ ./modules/dnn_superres/ ./modules/dnns_easily_fooled/ ./modules/dpm/ ./modules/face/ ./modules/freetype/ ./modules/fuzzy/ ./modules/hdf/ ./modules/hfs/ ./modules/img_hash/ ./modules/intensity_transform/ ./modules/julia/ ./modules/line_descriptor/ ./modules/matlab/ ./modules/mcc/ ./modules/ovis/ ./modules/phase_unwrapping/ ./modules/quality/ ./modules/rapid/ ./modules/README.md ./modules/reg/ ./modules/rgbd/ ./modules/saliency/ ./modules/sfm/ ./modules/shape/ ./modules/stereo/ ./modules/structured_light/ ./modules/superres/ ./modules/surface_matching/ ./modules/text/ ./modules/videostab/ ./modules/viz/ ./modules/wechat_qrcode/ ./modules/xfeatures2d/ ./modules/xobjdetect/ ./modules/xphoto/
COPY ./pythoncuda /root/opencv/opencv_contrib/modules/pythoncuda
WORKDIR /root/opencv
RUN mkdir -p build && cd build && cmake -DOPENCV_EXTRA_MODULES_PATH=/root/opencv/opencv_contrib/modules -DCMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON  -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages -D CUDA_ARCH_BIN=5.0,5.2,6.1,7.0,7.5,8.0,8.6 -DCUDA_ARCH_PTX=5.2 ../opencv-4.7.0/
RUN cd build && cmake --build . -j $(nproc)
RUN cd build && make install

RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install torch==1.13.1+cu117 \
                 torchvision==0.14.1+cu117 \
                 torchaudio==0.13.1+cu117 \
                 --extra-index-url https://download.pytorch.org/whl/cu117

# Install torch-scatter and torch-sparse matching torch 1.13.1+cu117
RUN pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
RUN pip3 install --no-deps git+https://github.com/argoverse/argoverse-api.git
RUN apt-get update && apt-get install -y libgl1
COPY ./ /root/CMRNext
WORKDIR /root/CMRNext
RUN pip3 install -r requirements.txt
WORKDIR /root/CMRNext/visibility_pkg
RUN python3 setup.py install

WORKDIR /root/
RUN git clone https://github.com/utiasSTARS/pykitti.git
RUN sed -i "s|print('Ground truth poses are not available for sequence ' +|pass|g" /root/pykitti/pykitti/odometry.py
RUN sed -i "s|self.sequence + '.')||g" /root/pykitti/pykitti/odometry.py
WORKDIR /root/pykitti
RUN python3 setup.py install

RUN apt-get update && apt-get install -y python3-tk

WORKDIR /
SHELL ["bash", "--command"]
ENV SHELL=/usr/bin/bash
CMD ["bash"]
