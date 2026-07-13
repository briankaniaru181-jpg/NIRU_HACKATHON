FROM python:3.11-slim

WORKDIR /app

COPY services/maktaba/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/maktaba/main.py .

COPY ["LITERARY WORKS", "./data/LITERARY WORKS"]

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]