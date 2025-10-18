FROM pytorch/pytorch:latest-cuda-runtime

WORKDIR /app
COPY requirements-vae.txt .
RUN pip install --no-cache-dir -r requirements-vae.txt

COPY main.py /app/main.py
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
