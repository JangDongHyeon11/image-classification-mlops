FROM python:3.9.17-slim

ARG EVIDENTLY_PORT=$EVIDENTLY_PORT

# Install system dependencies required for runtime and building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
        gcc \
        libc6-dev \
        && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip


WORKDIR /service

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE $EVIDENTLY_PORT

CMD evidently ui --workspace ${EVIDENTLY_WORKSPACE_NAME}/ --port ${EVIDENTLY_PORT}