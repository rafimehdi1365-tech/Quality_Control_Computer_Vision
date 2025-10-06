FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY requirements-preprocess.txt .
RUN pip install --no-cache-dir -r requirements-preprocess.txt

COPY preprocess.py /app/preprocess.py
CMD ["python","/app/preprocess.py"]
