#!/bin/bash

echo "Starting lightweight deployment..."

# Check for required tools
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker and Docker Compose are required."
    exit 1
fi

# Create necessary directories
mkdir -p mcp_server/data/documents
mkdir -p mcp_server/data/rulesets

# Build the frontend first
echo "Building the frontend..."
cd Frontend
npm ci
npm run build
cd ..

# Stop any running containers
echo "Stopping any existing containers..."
docker-compose down

# Start up the new containers
echo "Starting containers..."
docker-compose up -d

# Check if everything is running
echo "Checking containers..."
docker-compose ps

echo "Deployment complete! Access your application at http://localhost:8080"
echo "Note: If the application doesn't work immediately, allow a few moments for all services to start up."
echo "To see logs, run: docker-compose logs -f"
