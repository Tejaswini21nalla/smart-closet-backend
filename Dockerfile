# Use Python 3.9 slim as base image
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies including distutils
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache  # Clean pip cache

# Second stage: Runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=80

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "2", "--timeout", "120", "app:app"]