FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /logs && chmod 777 /logs

EXPOSE 5000

ENV PYTHONPATH=/app
ENV FLASK_APP=server.py
ENV FLASK_ENV=production
ENV LOG_FILE=/logs/app.log

CMD ["python", "server.py"]