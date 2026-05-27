FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY projects/bulldozer-price-regression ./projects/bulldozer-price-regression

ENTRYPOINT ["python", "projects/bulldozer-price-regression/src/train.py"]
