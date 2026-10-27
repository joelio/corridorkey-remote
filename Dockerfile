# CorridorKey Remote (unofficial) worker image for RunPod serverless.
#
# Build this yourself (RunPod GitHub integration or `docker build`). Do not
# push the result to a public registry: it contains the CorridorKey weights
# and the upstream licence forbids redistributing them (LICENSE, section 2a).
#
# Weights are baked in at build time so cold start measures container start
# and model load, not a 5 GB download.

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Pin upstream to an exact commit so timings are reproducible.
ARG CK_REPO=https://github.com/nikopueringer/CorridorKey2.0
ARG CK_COMMIT=485f5e9ab759ead6138cc990776184b59d6ea971

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CK_ROOT=/opt/corridorkey

# ffmpeg: Ubuntu's build includes libx264 and prores_ks, both used by
# upstream's writer (corridorkey/video_io.py:639-647). git is needed for the
# utils3d pin in upstream requirements.txt. libgl1/libglib2.0-0 for opencv.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR ${CK_ROOT}
RUN git init -q . \
 && git remote add origin "${CK_REPO}" \
 && git fetch -q --depth 1 origin "${CK_COMMIT}" \
 && git checkout -q FETCH_HEAD \
 && rm -rf .git

# torch 2.9.1 / torchvision 0.24.1 / CUDA 12.8 come from the base image.
# Pinning them in the same install stops pip from swapping torch out.
COPY worker/requirements.txt /tmp/worker-requirements.txt
RUN pip install -r requirements.txt -r /tmp/worker-requirements.txt torch==2.9.1 torchvision==0.24.1

# About 5 GB: CorridorKey v2 weights plus C-RADIOv4, MoGe-2 and RVM.
RUN python scripts/download_weights.py \
 && rm -rf /root/.cache/huggingface

COPY NOTICE THIRD_PARTY_NOTICES.md LICENSE /opt/ckremote/
COPY worker/handler.py /opt/ckremote/handler.py

# Fail the build, not the first paid job, if ffmpeg lacks an encoder we need.
RUN ffmpeg -hide_banner -encoders | grep -E ' (prores_ks|libx264) ' \
 && python scripts/smoke_test_runtime.py

CMD ["python", "-u", "/opt/ckremote/handler.py"]
