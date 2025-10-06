FROM python:3.10-slim
WORKDIR /app
COPY requirements-orch.txt .
RUN pip install --no-cache-dir -r requirements-orch.txt

COPY orchestrator.py /app/orchestrator.py
COPY configs /app/configs
EXPOSE 0
CMD ["python","/app/orchestrator.py"]
