# Third-Party Notices

This repository contains no third-party model weights or vendored code. The
worker image you build from it downloads the components below, so the image
does contain them. A copy of this file, `NOTICE` and `LICENSE` is placed in
`/opt/ckremote/` inside the image.

## CorridorKey 2.0

- Publisher: Corridor Digital LLC
- Source: https://github.com/nikopueringer/CorridorKey2.0 (pinned commit in
  `Dockerfile`), weights at https://huggingface.co/nikopueringer/CorridorKey2.0
- Licence: Corridor Key Licence, Version 2.0 (`LICENSE`)

CorridorKey's own `THIRD_PARTY_NOTICES.md` is in the image at
`/opt/corridorkey/THIRD_PARTY_NOTICES.md`, with the vendored MoGe and DINOv2
licences under `/opt/corridorkey/third_party/MoGe/LICENSE`.

## C-RADIOv4-SO400M

- Publisher: NVIDIA Corporation
- Source: https://huggingface.co/nvidia/C-RADIOv4-SO400M
- Licence: [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

## MoGe-2 ViT-B

- Publisher: Microsoft Corporation
- Source: https://github.com/microsoft/MoGe, weights at https://huggingface.co/Ruicheng/moge-2-vitb-normal
- Licence: MIT (code and weights). Vendored code includes DINOv2 (Meta
  Platforms, Inc.) under the Apache License 2.0.

## RVM (Recurrent Video MAE), small

- Publisher: Google DeepMind
- Source: https://github.com/google-deepmind/representations4d
- Licence: Apache License 2.0 (code), Creative Commons Attribution 4.0 (weights)

## Runtime libraries

- PyTorch base image `pytorch/pytorch` (BSD-3-Clause), which bundles NVIDIA
  CUDA and cuDNN runtime libraries under NVIDIA's own licence terms.
- FFmpeg from Ubuntu (LGPL/GPL, built with libx264 under GPL).
- `runpod` (MIT), `requests` (Apache 2.0), `boto3` (Apache 2.0).
