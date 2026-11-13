FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# LightGBM and XGBoost need the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a normal user, not root.
RUN useradd --create-home --uid 10001 app

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY configs ./configs
COPY src ./src

# Folders the project writes to; data/raw is where the Kaggle CSV is mounted.
RUN mkdir -p data/raw data/processed models reports logs \
    && chown -R app:app /app

USER app

HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD ["python", "-m", "src.health"]

CMD ["python", "-m", "src.train"]