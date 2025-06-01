# World-class Dockerfile for Python FastAPI + Celery + FFmpeg + Render.com
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY nr1-main/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY nr1-main/app ./app
COPY nr1-main/public ./app/public
COPY nr1-main/videos ./app/videos
COPY nr1-main/index.html ./app/index.html

ENV PYTHONUNBUFFERED=1
ENV ENV=production

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
