FROM anibali/pytorch:no-cuda

RUN sudo apt-get update && \
    sudo apt-get install -y\
    python-opencv\
    libzbar-dev\
  && sudo rm -rf /var/lib/apt/lists/*
COPY . /femida
WORKDIR /femida
RUN pip install --upgrade pip
RUN pip install -no-cache-dir -r requirements.txt -r requirements-dev.txt

RUN sudo chown -R user:user .
RUN pip install -e .
RUN py.test
RUN rm -rf tests/
