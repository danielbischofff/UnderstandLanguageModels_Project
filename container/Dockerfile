FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /workspace/
COPY src /workspace/src
RUN uv sync --frozen --no-dev

COPY main.py /workspace/main.py

CMD ["uv", "run", "python", "main.py"]
