mkdir -p "$UPLOAD_FOLDER"
mkdir -p "$RESULTS_FOLDER"
envsubst < /etc/nginx/conf.d/nginx.conf.template > /etc/nginx/conf.d/nginx.conf
chown nginx:nginx /etc/nginx/ssl
chmod 700 /etc/nginx/ssl
