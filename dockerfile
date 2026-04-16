# File: dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase
COPY . .

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "3000"]