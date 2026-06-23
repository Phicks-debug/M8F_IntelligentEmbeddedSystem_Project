FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY classification/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY classification/ ./classification/
RUN useradd -m -u 1000 runner && chown -R runner:runner /workspace
USER runner

ENV PYTHONUNBUFFERED=1
ENV HYDRA_FULL_ERROR=1
ENV STAGE=all

CMD python classification/src/main.py run --stage ${STAGE}
