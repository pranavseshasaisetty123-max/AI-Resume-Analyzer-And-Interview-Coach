FROM python:3.10-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY data/ /app/data/

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/analysis

# --- Backend Stage ---
FROM base AS backend
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Frontend Stage ---
FROM base AS frontend
EXPOSE 8501
ENV API_URL=http://backend:8000/api
CMD ["streamlit", "run", "src/web_app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
