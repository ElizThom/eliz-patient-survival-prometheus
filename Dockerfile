FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train/ train/
COPY main.py .
COPY xgboost-model.pkl .

EXPOSE 8080

CMD ["python", "main.py"]