# Timing columns

Every `ckremote run` appends one row to `timings.csv`. Times are in seconds.

| Column | Where it comes from |
| --- | --- |
| `upload_s` | Client: uploading the clip (and hint) to the bucket |
| `queue_delay_s` | RunPod `delayTime`: from submit until a worker picks the job up |
| `cold_start_s` | Same as `queue_delay_s` when the worker was cold, otherwise `warm`. Covers image pull, container start and model load |
| `model_load_s` | Worker: `load_model()` alone, cold runs only |
| `worker_download_s` | Worker: fetching the clip from the bucket |
| `inference_s` | Worker: upstream `run_inference()`, including its ffmpeg encode |
| `effective_fps` | `frames / inference_s` |
| `worker_upload_s` | Worker: putting outputs into the bucket |
| `download_s` | Client: fetching outputs |
| `wall_s` | Client: start of upload to end of download. Polls every 2 s, so up to 2 s too high |
| `billed_s_est` | RunPod `executionTime`, plus `queue_delay_s` on a cold run |
| `cost_usd_est` | `billed_s_est * rate_per_sec`, blank if no rate set |
| `hann_chunk`, `hann_stride`, `low_vram` | What upstream actually used, read from its settings log |
| `vram_budget_gb` | Free plus reserved GPU memory at job start, upstream's own formula. Should be close to the whole card |
| `peak_vram_gb` | `torch.cuda.max_memory_reserved()` during the job |

Cost is an estimate. RunPod's bill is the real figure.
