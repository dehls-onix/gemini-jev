FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies with light CPU-only torch
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers laya==0.3.4 a2a-sdk[http-server]==0.3.26 a2ui-agent-sdk==0.2.4

# Copy code
COPY . .

ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
