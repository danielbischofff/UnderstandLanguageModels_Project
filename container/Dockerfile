FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY container/requirements-linux.txt /tmp/requirements-linux.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://pypi.org/simple \
        -r /tmp/requirements-linux.txt \
    && rm -f /tmp/requirements-linux.txt

COPY main.py /workspace/main.py

CMD ["python", "main.py"]
