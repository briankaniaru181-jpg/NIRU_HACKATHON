FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY main.py .

# Expose port
EXPOSE 8004

# Start service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]