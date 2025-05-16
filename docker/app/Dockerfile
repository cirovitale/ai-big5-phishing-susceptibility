FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /logs && chmod 777 /logs

EXPOSE 5000

ENV PYTHONPATH=/app
ENV FLASK_APP=server.py
ENV FLASK_ENV=production
ENV LOG_FILE=/logs/app.log

CMD ["python", "server.py"]