FROM python:3.11-slim

# System deps: build tools for some wheels, and libs pdfplumber/pillow need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only torch first so sentence-transformers doesn't drag in the
# large CUDA wheels (keeps the image lean for the 4GB host).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY server/ ./server/
COPY config/ ./config/

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
