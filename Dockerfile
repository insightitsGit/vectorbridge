FROM python:3.11-slim

LABEL maintainer="Amin Parva <parvaamin@gmail.com>"
LABEL description="VectorBridge Agent — Universal vector database migration powered by CHORUS Fabric"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]"

# Copy source
COPY vectorbridge/ ./vectorbridge/

# Health check — pings license server
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f https://api.vectorbridge.io/v1/health || exit 1

# Agent entrypoint
CMD ["python", "-m", "vectorbridge.agent"]
