mkdir -p "$UPLOAD_FOLDER"
mkdir -p "$RESULTS_FOLDER"
envsubst < /etc/nginx/conf.d/nginx.conf.template > /etc/nginx/conf.d/nginx.conf
