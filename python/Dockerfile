FROM anibali/pytorch:no-cuda
RUN sudo apt-get update && \
    sudo apt-get install -y \
    python-opencv \
    libzbar-dev \
    netcat-traditional \
  && sudo rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -U \
    -r /tmp/requirements.txt \
    -r /tmp/requirements-dev.txt && \
    sudo rm /tmp/requirements.txt && \
    sudo rm /tmp/requirements-dev.txt
VOLUME /var/femida_detect
VOLUME /media
COPY . /femida_detect
WORKDIR /femida_detect
USER root
RUN pip install -e . --no-deps && py.test && rm -rf tests/
