# File: Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase
COPY . .

# Default command (can be overridden)
# Replace the CMD line at the bottom of your Dockerfile with:
CMD ["uvicorn", "server/main:app", "--host", "0.0.0.0", "--port", "6500"]
