# Stage 1: Build environment
FROM python:3.10-bullseye AS builder

WORKDIR /build

# Installa librerie di sistema necessarie
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    libhdf5-dev \
    libxkbcommon-x11-0 \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libgraphviz-dev \
    graphviz \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crea un virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia i requisiti
COPY requirements.txt .

# Installa i pacchetti Python (build da source per scipy, fix numpy compatibilità)
RUN pip uninstall -y keras keras-nightly keras-core || true \
    && pip install --upgrade pip setuptools wheel \
    && pip install numpy==1.26.4 \
    && pip install --no-binary=scipy scipy==1.15.3 \
    && pip install -r requirements.txt \
    && pip cache purge

# Pulizia dei file inutili dal virtualenv
RUN find /opt/venv -type f -name "*.pyc" -delete \
    && find /opt/venv -type f -name "*.pyo" -delete \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + || true \
    && find /opt/venv -type d -name "*.egg-info" -exec rm -rf {} + || true \
    && find /opt/venv -name "*.so" -exec strip {} \; || true \
    && find /opt/venv -name "*.a" -delete || true \
    && find /opt/venv -type f -name "*.md" -delete || true \
    && find /opt/venv -type f -name "*.txt" -delete || true \
    && find /opt/venv -type f -name "*.rst" -delete || true \
    && find /opt/venv -type d -name "tests" -exec rm -rf {} + || true \
    && find /opt/venv -type d -name "test" -exec rm -rf {} + || true \
    && find /opt/venv -type d -name "examples" -exec rm -rf {} + || true \
    && find /opt/venv -type d -name "docs" -exec rm -rf {} + || true


# Stage 2: Final runtime image
FROM python:3.10-bullseye

WORKDIR /app

# Installa solo librerie di runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi7 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenblas0 \
    libgfortran5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/*

# Copia il virtual environment costruito
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia i file dell'applicazione
COPY . .

# Pulizia dei file di sviluppo
RUN find . -type f -name "*.pyc" -delete \
    && find . -type d -name "__pycache__" -exec rm -rf {} + || true \
    && find . -type f -name "*.pyo" -delete \
    && find . -type d -name ".git" -exec rm -rf {} + || true \
    && find . -type f -name ".gitignore" -delete || true \
    && find . -type f -name "*.md" -delete || true \
    && find . -type f -name "Dockerfile" -delete || true

# Comando di default
CMD ["python", "server.py"]
