#!/usr/bin/env bash
mkdir -p "$UPLOAD_FOLDER"
mkdir -p "$RESULTS_FOLDER"
python configure_nginx.py
python configure_mongo.py
