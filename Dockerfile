FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY requirements-serving.txt .
RUN python -m pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY data ./data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/live', timeout=3)"

CMD ["python", "-m", "uvicorn", "src.ecommerce_llm.app:app", "--host", "0.0.0.0", "--port", "8000"]
