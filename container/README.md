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

The first image is intentionally small: it contains Python and PyTorch only. We can add Transformers, Accelerate, and the project dependencies after the container runs successfully on the cluster.
