FROM python:3.12-alpine
RUN pip install --no-cache-dir pymongo==4.18.3
COPY seed.py /seed.py
CMD ["python", "-I", "/seed.py"]
