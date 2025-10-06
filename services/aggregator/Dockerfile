FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
COPY requirements-agg.txt .
RUN pip install --no-cache-dir -r requirements-agg.txt

COPY aggregator.py /app/aggregator.py
EXPOSE 0
CMD ["python","/app/aggregator.py"]
