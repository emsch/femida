FROM tiangolo/uwsgi-nginx-flask:python3.6-alpine3.7
MAINTAINER Maxim Kochurov

COPY app/requirements-frontend.txt /tmp/requirements-frontend.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -U -r /tmp/requirements-frontend.txt && \
    rm /tmp/requirements-frontend.txt
VOLUME /media
COPY nginx/femida.conf /etc/nginx/conf.d/nginx.conf.template
COPY nginx/ /etc/nginx/
COPY app /app

EXPOSE 80 443
