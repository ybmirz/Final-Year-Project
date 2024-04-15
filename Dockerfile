FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /experiments

# Add the NVIDIA public GPG key
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y \
    build-essential supervisor git cmake vim

ADD experiments /experiments
RUN pip install --upgrade setuptools
RUN pip install --upgrade importlib_metadata
# Upgrade Cython
RUN pip install --upgrade cython
RUN pip install -r requirements.txt

ADD tensorboard.conf /etc/supervisor/conf.d/tensorboard.conf

CMD supervisord -n
