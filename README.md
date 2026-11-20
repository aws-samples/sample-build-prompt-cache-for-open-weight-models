# How to use prompt caching for LLMs on Amazon SageMaker AI

Companion code for the AWS blog post **"How to use prompt caching for LLMs on Amazon SageMaker AI."**

It deploys DeepSeek-R1-Distill-Qwen-1.5B and -7B to SageMaker AI real-time endpoints (one `ml.g5.xlarge`,
one GPU with 24 GB GPU memory, each) using the **AWS SGLang Deep Learning Container**, and measures what SGLang's RadixAttention
prefix cache does for latency, throughput, and cost, with the cache on and off.

| File | What it is |
|---|---|
| [`sagemaker_sglang_prompt_caching.ipynb`](sagemaker_sglang_prompt_caching.ipynb) | Step-by-step notebook (SageMaker Studio, CPU kernel) |
| [`run_benchmark.py`](run_benchmark.py) | All logic: `deploy`, `kvpool`, `bench`, `remote`, `analyze`, `cleanup` |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Command-line reproduction, configuration, troubleshooting |
| [`results/`](results/) | Raw per-request CSVs, summary tables, and figures from the published run |

## Headline results (us-east-1, SGLang 0.5.19 DLC, client in-Region)

Shared prefix is measured in **tokens**. Warm requests reuse the prefix with a **different user query** each time.

| Model | Prefix | TTFT cold → warm | End-to-end (32 output tokens) cold → warm |
|---|---|---|---|
| 1.5B | 2,048 tok | 150 → 35 ms (−77%) | 389 → 273 ms (−30%) |
| 7B | 2,048 tok | 530 → 73 ms (−86%) | 1,536 → 1,078 ms (−30%) |
| 7B | 8,192 tok | 2,093 → 76 ms (−96%) | 3,122 → 1,105 ms (−65%) |

Under load (4 × 2,048-token prefixes, new query per request, 128 output tokens, one instance):

| Model | Peak throughput cache on / off | $ per 1M requests at peak, on / off |
|---|---|---|
| 1.5B | 12.5 / 4.9 req/s | $31 / $81 |
| 7B | ≥ 6.3 / 1.6 req/s | $62 / $251 |

KV pool on `ml.g5.xlarge`: 560,645 tokens (1.5B) and **77,314 tokens (7B)**. With more than ~9 distinct 8K-token prefixes in
rotation, the 7B hit rate falls from 99% to 0% (`results/fig4_eviction.png`).

## Cost

Each endpoint is $1.408/hr (us-east-1, on-demand, at time of writing; see [SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/)). The published run used four endpoints for about 50 minutes (≈ $5)
plus a small Processing job for the client. Run `python run_benchmark.py cleanup` (or the notebook's final cell).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
