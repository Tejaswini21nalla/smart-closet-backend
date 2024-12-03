# Use Python 3.9 Alpine for a smaller image
FROM python:3.9-alpine as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache --virtual .build-deps build-essential python3-dev python3-distutils

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Second stage: Runtime
FROM python:3.9-alpine

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

# Expose port
EXPOSE 80

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "2", "--timeout", "120", "app:app"]