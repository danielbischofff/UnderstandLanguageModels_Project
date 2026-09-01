# Container setup

Container definitions and build instructions for the experiment environment.

## Build

Build this image on the Linux cluster, or explicitly target Linux from another machine:

```bash
docker build --platform linux/amd64 \
  -f container/Dockerfile \
  -t ullm-project:dev .
```

## Test

```bash
docker run --rm ullm-project:dev
```

The image installs the locked uv environment, including Transformers, Accelerate,
and the steering package. For a CUDA-specific PyTorch build or bitsandbytes,
build from the cluster's CUDA base image and run `uv sync --extra quantization`.
