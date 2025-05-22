FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for the frontend
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Build the frontend
WORKDIR /app/Frontend
RUN npm ci && npm run build

# Back to app directory
WORKDIR /app

# Expose ports for the services
EXPOSE 8000 8050 8080

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8050
ENV PYTHONUNBUFFERED=1

# Create a start script
RUN echo '#!/bin/bash\n\
python -m mcp_server.server &\n\
python -m mcp_server.mainWeb &\n\
caddy run --config /app/Caddyfile\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Run the start script
CMD ["/app/start.sh"]
