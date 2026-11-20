# Deployment guide: How to use prompt caching for LLMs on Amazon SageMaker AI

One path: SGLang served by the **AWS SGLang Deep Learning Container (SageMaker variant)** on SageMaker AI
real-time endpoints. No Docker host or EC2 instance is needed.

| Item | Value |
|---|---|
| Image | `763104351884.dkr.ecr.<region>.amazonaws.com/sglang:server-sagemaker-cuda-v1.4.1` |
| Instance | `ml.g5.xlarge` (one GPU, 24 GB GPU memory) per endpoint |
| Host AMI | `InferenceAmiVersion=al2023-ami-sagemaker-inference-gpu-4-1` (required for the CUDA 13 image) |
| Models | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

## Prerequisites

- A SageMaker execution role (the notebook uses `sagemaker.get_execution_role()`, or set `SM_EXEC_ROLE`).
- Service quota **`ml.g5.xlarge for endpoint usage` ≥ 4** for the full run (cache on + cache off for two models).
  With 1–2 available, set `SIZES = ["1.5b"]` and/or `RUN_CONTROL = False` in the notebook.
- Python 3.10+ with `boto3`, `tokenizers`, `huggingface_hub`, `pandas`, `matplotlib` (the first notebook cell installs them).
- Run the client **in the endpoint's Region** (SageMaker Studio JupyterLab, a notebook instance, or the `remote`
  sub-command below). TTFT is measured client-side, so a distant client adds WAN latency to every number.

## Option 1 — notebook

Open `sagemaker_sglang_prompt_caching.ipynb` in SageMaker Studio (a CPU kernel is enough) next to
`run_benchmark.py`, and **Run All**. The last cell deletes everything it created.

## Option 2 — command line

```bash
ROLE=arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole>
BUCKET=<an S3 bucket in the same Region>

# 1. Deploy (about 8-10 min). Cache ON is SGLang's default; the OFF endpoint is the negative control.
for s in 1.5b 7b; do for c in on off; do
  python run_benchmark.py deploy --size $s --cache $c --role $ROLE --suffix v10 --no-wait
done; done
aws sagemaker wait endpoint-in-service --endpoint-name pc-7b-nocache-v10   # repeat per endpoint

# 2. KV-pool size from the SGLang start-up log (CloudWatch Logs)
python run_benchmark.py kvpool \
  --endpoint pc-15b-cache-v10 pc-15b-nocache-v10 pc-7b-cache-v10 pc-7b-nocache-v10 \
  --labels   1.5b-cache       1.5b-nocache       7b-cache       7b-nocache

# 3a. Benchmark from a machine in the same Region ...
python run_benchmark.py bench --endpoint pc-7b-cache-v10 --size 7b --label 7b-cache --experiments sweep,load,evict

# 3b. ... or from anywhere: run the client in a SageMaker Processing job (ml.m5.xlarge) in the Region
python run_benchmark.py remote --role $ROLE --bucket $BUCKET \
  --job pc-15b-cache-v10:1.5b:1.5b-cache:sweep,load,evict \
  --job pc-15b-nocache-v10:1.5b:1.5b-nocache:sweep,load \
  --job pc-7b-cache-v10:7b:7b-cache:sweep,load,evict \
  --job pc-7b-nocache-v10:7b:7b-nocache:sweep,load
aws s3 cp --recursive s3://$BUCKET/prompt-cache-blog/<run-id>/results/ results/

# 4. Tables + figures + cost
python run_benchmark.py analyze

# 5. Clean up (only resources named pc-* AND tagged project=prompt-cache-blog)
python run_benchmark.py cleanup
```

## What the benchmark does

| Experiment | Design | Output |
|---|---|---|
| `sweep` | Prefix 512–8,192 **tokens** (exact, via the model tokenizer). 5 independent prefixes per length; each gives 1 cold request (random nonce at the start of the prefix forces a miss) + 4 warm requests with **different user queries**. 32 output tokens, concurrency 1. | `<label>_sweep.csv`, `table_sweep.csv`, Fig. 1–2 |
| `load` | Closed loop at 1/4/8/16/32 concurrent clients. 4 distinct 2,048-token prefixes, fresh query per request, 128 output tokens. Cache on vs. off. | `<label>_load.csv`, `table_load.csv`, Fig. 3 |
| `evict` | Round-robin over N = 4…24 distinct 8,192-token prefixes, 3 passes, hit rate on passes 2–3. | `<label>_evict.csv`, `table_evict.csv`, Fig. 4 |

Every request records client-side TTFT, end-to-end latency, `prompt_tokens`, and `cached_tokens`
(`usage.prompt_tokens_details.cached_tokens`, enabled with `SM_SGLANG_ENABLE_CACHE_REPORT=true`).

## Configuring SGLang on SageMaker

Any `SM_SGLANG_<FLAG>` environment variable becomes `--<flag>` (lower-case, `_` → `-`; `true` → bare flag).
Useful ones for prefix caching:

| Variable | Effect |
|---|---|
| `SM_SGLANG_DISABLE_RADIX_CACHE=true` | turn prefix caching off |
| `SM_SGLANG_RADIX_EVICTION_POLICY=lfu` (or `slru`) | keep frequently reused prefixes under memory pressure (default `lru`) |
| `SM_SGLANG_MEM_FRACTION_STATIC=0.85` | GPU memory for weights + KV pool |
| `SM_SGLANG_KV_CACHE_DTYPE=fp8_e5m2` | roughly double the KV pool (validate quality for your model) |
| `SM_SGLANG_ENABLE_HIERARCHICAL_CACHE=true` | HiCache: spill evicted KV to host memory |

## Troubleshooting

- **First request after `InService` is slow or times out** – the first call JIT-compiles kernels (~9 s observed).
  The benchmark sends three warm-up requests before measuring.
- **Endpoint fails health check** – confirm `InferenceAmiVersion` is set; the default host AMI drivers are too old for CUDA 13.
- **`/flush_cache` / `/metrics` not reachable** – SageMaker only forwards `/invocations` and `/ping`; that's why cold
  requests use a fresh prefix instead of a cache flush.
