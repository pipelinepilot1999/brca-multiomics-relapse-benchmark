# Multi-omics biomarker signature pipeline.
#
# Build on x86_64. On Apple Silicon you MUST cross-build:
#     docker build --platform linux/amd64 -f docker/Dockerfile -t biomarker-brca:1.0 .
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="biomarker-signature-brca" \
      org.opencontainers.image.description="Multi-omics biomarker signature for breast cancer relapse" \
      org.opencontainers.image.version="1.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    MPLBACKEND=Agg

# procps supplies `ps`, which Nextflow needs to collect task metrics.
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pipeline

COPY env/requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY scripts/ ./scripts/
COPY config/ ./config/
COPY nextflow/ ./nextflow/

# Fail the build if the environment cannot import what the pipeline needs.
RUN python -c "import pandas, numpy, sklearn, xgboost, torch, shap, lifelines, gseapy, reportlab; \
print('container imports OK; torch', torch.__version__)"

ENV PYTHONPATH=/opt/pipeline/scripts
WORKDIR /work
ENTRYPOINT ["python"]
CMD ["-c", "print('biomarker-signature-brca container. Mount the project at /work.')"]
