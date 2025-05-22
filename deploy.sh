#!/bin/bash

# Production Deployment Script for MCP Server
set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting MCP Server deployment...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if Tailscale is installed
if ! command -v tailscale &> /dev/null; then
    echo -e "${YELLOW}Tailscale is not installed. You may want to install it for secure remote access.${NC}"
    read -p "Continue without Tailscale? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# # Pull latest changes if this is a git repository
# if [ -d .git ]; then
#     echo -e "${GREEN}Pulling latest changes from git...${NC}"
#     git pull
# fi

# Ensure data directories exist
echo -e "${GREEN}Setting up data directories...${NC}"
mkdir -p mcp_server/data/documents
mkdir -p mcp_server/data/rulesets

# Copy production Caddyfile
echo -e "${GREEN}Setting up Caddyfile for production...${NC}"
# cp Caddyfile Caddyfile

# Build and start the containers
echo -e "${GREEN}Building and starting containers...${NC}"
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Setup Ollama model (if not already loaded)
echo -e "${GREEN}Setting up Ollama models...${NC}"
sleep 10 # Give Ollama container time to start
if ! docker exec ollama ollama list | grep -q llama3.2; then
    echo -e "${YELLOW}Downloading Llama3.2 model (this may take a while)...${NC}"
    docker exec ollama ollama pull llama3.2
fi

# Check if services are running
echo -e "${GREEN}Checking if services are running...${NC}"
sleep 5

if curl -s http://localhost:8080 > /dev/null; then
    echo -e "${GREEN}Frontend is running!${NC}"
else
    echo -e "${RED}Frontend is not responding. Check logs with 'docker-compose logs'.${NC}"
fi

if curl -s http://localhost:8000/docs > /dev/null; then
    echo -e "${GREEN}API server is running!${NC}"
else
    echo -e "${RED}API server is not responding. Check logs with 'docker-compose logs'.${NC}"
fi

# Get Tailscale info if available
if command -v tailscale &> /dev/null; then
    echo -e "${GREEN}Tailscale information:${NC}"
    TAILSCALE_IP=$(tailscale ip -4)
    echo -e "Your Tailscale IP: ${YELLOW}$TAILSCALE_IP${NC}"
    echo -e "Access your application at: ${YELLOW}http://$TAILSCALE_IP:8080${NC}"
fi

echo -e "${GREEN}Deployment completed!${NC}"
echo -e "To check logs, run: ${YELLOW}docker-compose logs -f${NC}"
echo -e "To stop the services, run: ${YELLOW}docker-compose down${NC}"
