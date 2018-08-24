mkdir -p docker/ssl/
openssl req -new -x509 -days 9999 -nodes -out docker/ssl/cert.pem -keyout docker/ssl/cert.key
chown nginx:nginx /etc/nginx/ssl
chmod 700 docker/ssl/

