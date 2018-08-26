FROM anibali/pytorch:no-cuda

RUN sudo apt-get update && \
    sudo apt-get install -y\
    python-opencv\
    libzbar-dev\
  && sudo rm -rf /var/lib/apt/lists/*
COPY python/requirements.txt /tmp/requirements.txt
COPY python/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    -r requirements.txt \
    -r requirements-dev.txt \
    rm /tmp/requirements.txt \
    rm /tmp/requirements-dev.txt

COPY python /femida
WORKDIR /femida
RUN sudo chown -R user:user .
RUN pip install -e . --no-deps && py.test && rm -rf tests/
