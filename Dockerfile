FROM public.ecr.aws/docker/library/python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
   build-essential curl \
   && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]


