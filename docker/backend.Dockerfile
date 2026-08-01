# Packaging only (Docker Milestone) -- does not modify anything under
# production/src/serving/. Builds the FastAPI/uvicorn live-serving layer
# (api.py) as a standalone container.
FROM python:3.11-slim

WORKDIR /app

# opencv-python (imported transitively via production.src.cv, which api.py
# imports at module level for the source="cv" WebSocket path) needs these
# shared libs even though this container never opens a display.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# requirements.txt lists "torch" unpinned; PyPI's default Linux wheel for
# that bundles the full NVIDIA CUDA runtime (several GB) even though this
# container has no GPU and never will. Installing the CPU-only wheel from
# PyTorch's own index FIRST satisfies that requirement before the
# unconstrained requirements.txt install would otherwise pull the much
# larger default -- this alone was enough to exhaust this host's disk
# during the first build attempt.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY production/ production/

# Belt-and-suspenders: api.py already sets this itself via
# `os.environ.setdefault` (the standalone-launch fix from Milestone 17),
# but setting it here too means the container's declared environment is
# self-documenting without relying on that line staying in the source.
ENV MLFLOW_ALLOW_FILE_STORE=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "production.src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
