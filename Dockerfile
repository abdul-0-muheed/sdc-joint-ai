# syntax=docker/dockerfile:1

# Use Python 3.13 slim as base image
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim-bookworm

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install build dependencies for native Python packages
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /sdc-joint-ai

# Copy requirements first for better layer caching
COPY requirement.txt .

# Create virtual environment and install dependencies
RUN python -m venv venv \
  && . venv/bin/activate \
  && pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirement.txt

# Copy all application files
COPY . .

# Activate venv and run the agent
CMD ["venv/bin/python", "src/agent.py", "start"]
