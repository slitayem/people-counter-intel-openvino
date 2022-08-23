FROM openvino/ubuntu20_dev:2021.4.2_20210416
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE=/home/openvino/people-counter

ENV CONDA_VERSION=latest
ENV OS_TYPE=x86_64
ENV PYTHON_VERSION=3.8
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino
USER root

RUN apt-get -yqq update \ 
    && apt-get install -yq --no-install-recommends \
      git \
      yasm libx264-dev \
      vim \
      make \
      g++ \
      wget \
      npm \
      libzmq3-dev \
      libkrb5-dev \
      ffmpeg \
    && apt-get clean \  
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt $WORKSPACE/
RUN python3 --version \
    && python3 -m pip install --upgrade pip \
    && pip3 --no-cache-dir install -r $WORKSPACE/requirements.txt \
    && pip3 --no-cache-dir install -r ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements.txt

###### Configure the Docker Image with access to GPU
# https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_docker_linux.html
# Used image version https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_docker_linux.html
# Use `20.35.17767` for 10th generation Intel® Core™ processor (formerly Ice Lake) 
# or 11th generation Intel® Core™ processor (formerly Tiger Lake)
ARG INTEL_OPENCL=19.41.14441
ARG DEVICE
RUN if [[ "${DEVICE,,}" == "gpu" ]] ; then \
  ${INTEL_OPENVINO_DIR}/install_dependencies/install_NEO_OCL_driver.sh --no_numa -y --install_driver ${INTEL_OPENCL};\
    rm -rf /var/lib/apt/lists/* ; \
  fi

WORKDIR ${WORKSPACE}
COPY src ${WORKSPACE}/src
COPY webservice ${WORKSPACE}/webservice
RUN ls ${WORKSPACE}

# Upgrade npm
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash \
    && apt-get install -y nodejs \
    &&  npm -v \
    && npm update npm -g \
    && npm -v

#  http://www.tiernok.com/posts/2019/faster-npm-installs-during-ci/
RUN cd $WORKSPACE/webservice/server \
  && cp .npmrc $HOME/ \ 
  && npm install -g npm \
  && npm init --yes \
  && npm install \
  && npm dedupe \
  && cd $WORKSPACE/webservice/ui \
  && npm cache clean --force \
  && npm install \
  && npm dedupe \
  && rm -rf /tmp/npm* \
  && chown -R openvino $WORKSPACE

# We need to check out an old version of FFmpeg to have FFserver as FFmpeg no longer bundles it
RUN git clone -c http.sslverify=false https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg \
  && cd /tmp/ffmpeg \
  && git checkout 2ca65fc7b74444edd51d5803a2c1e05a801a6023 \
  && ./configure \
  && make \
  && make install \
  && rm -rf /tmp/ffmpeg

USER openvino
RUN echo "source ${INTEL_OPENVINO_DIR}/bin/setupvars.sh -pyver ${PYTHON_VERSION}" >> $HOME/.bashrc

WORKDIR $WORKSPACE
EXPOSE 3000 3002 3004

CMD /bin/bash