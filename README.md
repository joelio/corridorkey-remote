# CorridorKey Remote (unofficial)

> **Archived, October 2026.** This experiment stopped before any GPU timings were taken. The code and licence notes are left here for reference. For running CorridorKey without a local NVIDIA card, see [Related projects](#related-projects).

**This is not made by, endorsed by or affiliated with Corridor Digital.** It is a free, community bridge to their open [CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0) model. If you can run CorridorKey locally, or you want the supported plugin, go to [corridorkey.com](https://corridorkey.com).

CorridorKey 2.0 needs an NVIDIA RTX 30-series card or newer. This project lets you run it on a GPU you rent yourself on [RunPod](https://www.runpod.io) serverless, from any machine: an AMD card, Linux, an Intel Mac, a laptop. You send a clip and get the matte and foreground back.

It is also an experiment. The main thing it produces is a timing table: is a cloud round trip fast enough to be useful? See [Results](#results).

## How it works

```
your machine                 your bucket (R2)            your RunPod endpoint
------------                 ----------------            --------------------
ckremote run shot.mov  --->  shot.mov          --->      download, infer.py runtime,
                                                         upload alpha.mp4, fg.mp4
outputs downloaded     <---  alpha.mp4, fg.mp4  <---
timings appended to results/timings.csv
```

- The worker runs upstream's own `load_model()` and `run_inference()` code unchanged, at a pinned commit.
- Weights are baked into the image when you build it, so a cold start never waits on a 5 GB download.
- The client gives the worker presigned links, so your bucket keys never leave your machine.
- Everything runs on your accounts. Nothing passes through us, and we never see your footage or your bill.

## What it costs

You pay RunPod per second of GPU time, including the start-up time of a cold worker, and your storage provider for what you store. We do not charge anything and never will.

- GPU prices: [RunPod serverless pricing](https://www.runpod.io/pricing)
- Storage: [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/). R2 has a free tier and no download fees, which is why we use it. `ckremote` deletes each job's files from the bucket when it finishes.

Set `CKREMOTE_RATE_PER_SEC` to your GPU's price and every run prints an estimated cost.

## Setup

You need a GitHub account, a RunPod account with credit, a Cloudflare account and Python 3.10 or newer. You do not need Docker or an NVIDIA card.

1. **Fork this repository** on GitHub.
2. **Create a RunPod API key** under Settings, API Keys.
3. **Create the endpoint.** In the RunPod console go to Serverless, New Endpoint, and choose your fork as the GitHub source, with Dockerfile path `Dockerfile`. Pick one or more GPU types with at least 16 GB, set max workers to 1, container disk to at least 20 GB and execution timeout to 1800 seconds. The first build takes around 20 to 30 minutes, mostly downloading weights. If it does not start, create a release on your fork; RunPod builds on releases.
4. **Create an R2 bucket** in Cloudflare, then an R2 API token with Object Read and Write on that bucket only. Note the access key ID, secret and the S3 endpoint (`https://<account-id>.r2.cloudflarestorage.com`).
5. **Install the client:**
   ```
   git clone https://github.com/<you>/<your-fork>.git ckremote && cd ckremote
   python3 -m venv .venv && . .venv/bin/activate
   pip install .
   ```
6. **Configure it:** `cp .env.example .env`, then fill in `.env`. Git ignores this file.
7. **Check it:** `ckremote check` confirms the bucket and endpoint work before you spend any GPU time.
8. **Key a clip:**
   ```
   ckremote run shot.mov
   ```
   The outputs land in `shot_ckremote/` next to the clip, and the timings go into `results/timings.csv`.

### Options

```
ckremote run shot.mov --hint frame1_matte.png     # first-frame hint (white = subject)
ckremote run shot.mov --outputs alpha fg cutout   # cutout is ProRes 4444 with alpha
ckremote run shot.mov --hann 48/24                # override the automatic window
ckremote run shot.mov --label 4090-4k-30s         # tag the CSV row
ckremote run tiny.mp4 --inline                    # no bucket, clips under 7 MB only
```

The outputs, hint and Hann options mean what they mean in upstream's README. With no `--hann`, upstream picks the largest window that fits the card's free memory.

### Please do not publish the image

Build the image in your own RunPod account (step 3) or your own private registry. Do not push it to a public registry. It contains the CorridorKey weights, and the licence does not allow redistributing them (section 2a). Upstream itself does not redistribute the three foundation models; its download script fetches them from their publishers.

## Results

Not measured yet. The plan:

| GPU | VRAM | 1080p 5 s | 1080p 30 s | 4K 5 s | 4K 30 s |
| --- | --- | --- | --- | --- | --- |
| RTX 4090 | 24 GB | | | | |
| L40S or A6000 | 48 GB | | | | |
| Cheapest working 16 GB card | 16 GB | | | | |

Each cell will record cold and warm runs: upload, cold start, model load, inference, effective fps, download, wall clock and estimated cost. The raw rows are in [`results/timings.csv`](results/timings.csv), and [`results/README.md`](results/README.md) explains the columns.

**Is it fast enough to use?** We will answer this honestly once the numbers are in.

## Related projects

Other ways to run CorridorKey without a local NVIDIA card, as of October 2026. Check them before you rent a GPU.

- [corridorkey.cloud](https://corridorkey.cloud) ([source](https://github.com/DCRepublic/CorridorKey_web)): a hosted community render farm where you contribute GPU time to earn processing credits. Built on CorridorKey 1.
- [gercheg/corridorkey-runpod-worker](https://github.com/gercheg/corridorkey-runpod-worker): a RunPod serverless worker for CorridorKey 1, with a prebuilt public image.
- [CorridorKey-Runtime](https://github.com/alexandremendoncaalvaro/CorridorKey-Runtime): a native runtime and Resolve and Nuke OFX plugin built with Corridor Digital, with an experimental DirectML track for AMD and Intel cards on Windows.
- [corridorkey-desktop](https://github.com/MarcelineVPQ/corridorkey-desktop): CorridorKey 2.0 rewritten on Rust and ONNX Runtime, about twice as fast as upstream. NVIDIA only for now, with Apple Silicon in progress.
- [corridorkey-mlx](https://github.com/nikopueringer/corridorkey-mlx) and [CorridorKeyOpenVINO](https://github.com/daniil-lyakhov/CorridorKeyOpenVINO): ports for Apple Silicon and for Intel hardware.

What this project adds: CorridorKey 2.0, on hardware you rent and control, with the timings published.

## Licence

The Corridor Key Licence, Version 2.0, the same licence as CorridorKey 2.0: a variant of CC BY-NC-SA 4.0 with extra terms. See [`LICENSE`](LICENSE) (copied unchanged from upstream), [`NOTICE`](NOTICE) for what this project adds, and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

In short:

- This project is free and stays free. It must not be sold, put behind a paywall or run as a paid API.
- You can use what you make with it commercially. Your footage and mattes are yours.
- Organisations with more than US $20 million in annual revenue need a CorridorKey plugin seat for each person using it, whether locally or through this project (licence section 3).
- The foundation models keep their own licences. NVIDIA's C-RADIO licence allows commercial use and redistribution, but NVIDIA can revoke it if you sue over patents or copyright in the model, NVIDIA can update the terms, and anyone who redistributes the model must include NVIDIA's licence and notice. That last point is another reason not to publish the image.

## Development

```
pip install -e ".[dev]" pre-commit
pre-commit install
python -m pytest
```

The tests run on a CPU with upstream stubbed out. Pre-commit and CI run secret scanning (gitleaks), ruff, hadolint, actionlint and the tests on every change, so mistakes show up before a slow and costly GPU run.

Upstream notes, things we found reading their code (not bugs we have raised):

- A clip named `input.*`, `frames.*` or `rgb.*` takes its output names from the folder it sits in (`corridorkey/video_io.py:58-63`). The worker handles this.
