# Changed the PyTorch base image since Pennylane requires > Python 3.10
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
WORKDIR /experiments
# Add the NVIDIA public GPG key
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y \
    build-essential supervisor git cmake vim

ADD experiments /experiments

RUN pip install --upgrade setuptools
RUN pip install --upgrade importlib_metadata

RUN git clone https://github.com/sunqm/libcint.git && \
    cd libcint && \
    mkdir build && cd build && \
    cmake .. && \
    make && \
    make install

# Set the environment variable for pyscf to find libcint
ENV PYSCF_INC_DIR=/usr/local/lib

# # Upgrade Cython
RUN pip install --upgrade cython
# RUN pip install --upgrade qulacs-gpu
# RUN pip install --upgrade scikit-learn
# RUN pip install --upgrade scipy
# Install pyscf
RUN pip install -r requirements.txt --use-pep517

ADD tensorboard.conf /etc/supervisor/conf.d/tensorboard.conf

CMD supervisord -n
