FROM tensorflow/tensorflow:1.15.4-gpu-py3-jupyter

ARG DEBIAN_FRONTEND=noninteractive

# set time zone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update
RUN apt-get -y install git

RUN pip install --upgrade pip

WORKDIR /inpainting

# Python dependencies
COPY requirements ./requirements

RUN pip install -r requirements/base.txt

# copy access token
COPY .netrc /root/.netrc

# gRPC API
RUN pip install git+https://bitbucket.synergeticon.com/scm/pde/protodef.git
