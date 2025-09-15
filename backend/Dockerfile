FROM python:3.10-slim

WORKDIR /app

#Optional system packages for audio/TensorFlow libs
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (expects backend/requirements.txt in this folder)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source (expects this Dockerfile to be in backend/)
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]



