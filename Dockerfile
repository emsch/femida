FROM python:3.6
RUN apt-get update && apt-get install \
    build-essential\
    libzbar-dev\
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY . /app
WORKDIR /app

RUN cp Makefile.template Makefile
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
RUN pip install -e .
RUN make tests

