# xing4_0 Inference Service Usage Guide

> Image: `harbor.telecom-ai.com.cn/library/vllm-ascend:v0.26.0rc1-29b-xing4_0`
> Scope: Deployment, startup, and testing of OpenAI-compatible inference services based on vLLM-Ascend

---

## 1. Overview

This document describes how to deploy the **Xing4.0** large language model using the vLLM-Ascend inference framework on Ascend NPU environments.

Overall workflow:

1. Load the image and start the container;
2. Prepare the main model weights (draft model weights are only required for dspark speculative decoding mode);
3. Start the OpenAI-compatible inference service (`/v1/completions` and other endpoints);
4. Verify via `curl` or the OpenAI SDK.

---

## 2. Image and Model Overview

| Item | Content |
| --- | --- |
| Open-source image | `harbor.telecom-ai.com.cn/library/vllm-ascend:v0.26.0rc1-29b-xing4_0` |
| Offline image file | `/data01/workspace/lzx/llm/images/xing4_0-29b-v26.tar.gz` |
| Inference framework | vLLM-Ascend `v0.26.0rc1` |
| Model scale | 29B (main model) |
| Exposed service name | `xingchen4` (set via `--served-model-name`) |

This image comes with a vLLM-Ascend environment optimized for Xing4.0, with several acceleration features enabled (see Section 7 for detailed startup parameters).

---

## 3. Environment and Resource Requirements

- **Hardware**: Ascend **910B2** series NPUs, at least **2** cards (the startup script uses `--tensor-parallel-size 2` for 2-card tensor parallelism); the reference script passes through 8 cards (`/dev/davinci0` ~ `/dev/davinci7`), and the service uses any 2 of them as needed;
- **Software**: The host must have Ascend driver CANN runtime compatible with the image installed, with `davinci` device nodes correctly mounted;
- **Storage**:
  - Main model weights directory (`$weight_path` in the scripts), required for both startup modes;
  - Draft model weights directory (only required for dspark speculative decoding): `/hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_8_8/checkpoints/checkpoint_best`. The mtp mode does not require a draft model; see Section 6.

---

## 4. Image Loading and Container Startup

### 4.1 Loading the Offline Image

```bash
docker load -i /data01/workspace/lzx/llm/images/xing4_0-29b-v26.tar.gz
```

Confirm the image exists after loading:

```bash
docker images | grep vllm-ascend
```

### 4.2 Starting the Container (Reference Script)

Ascend containers need NPU devices and the driver directory passed through. Reference script as follow (`$1` is the container name, `$2` is the image name):

```bash
docker run -itd -u 0 \
    --name $1 \
    --net=host \
    --privileged=true \
    --shm-size=512g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /mnt/sfs_turbo/.cache:/root/.cache \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /var/log/npu/:/usr/slog \
    -e VLLM_USE_V1=1 \
    -it $2 /bin/bash
```

Example (`$1` = container name, `$2` = image name):

```bash
docker run -itd -u 0 --name xing4_0-vllm --net=host --privileged=true --shm-size=512g \
  [--device and -v parameters as in the script above] \
  -e VLLM_USE_V1=1 -it harbor.telecom-ai.com.cn/library/vllm-ascend:v0.26.0rc1-29b-xing4_0 /bin/bash
```

> Notes:
> - This script is designed for **Ascend 910B2** environments and passes through `/dev/davinci0` ~ `/dev/davinci7` (a total of **8** NPUs). The service uses 2 of them via `--tensor-parallel-size 2`;
> - `--net=host` uses the host network; the service port (`--port`, dspark=8000 / mtp=8009) binds directly to the host, so no `-p` port mapping is needed;
> - `-u 0` runs as root; `--privileged=true` and `--shm-size=512g` are common settings for Ascend inference containers;
> - Add `-v` mounts for business directories (e.g., weights) as needed;
> - `-e VLLM_USE_V1=1` enables the vLLM V1 execution engine.

---

## 5. Model Weight Preparation

1. Place the main model weights in a host directory and pass them via `$weight_path` in the startup script, e.g.:

   ```bash
   export weight_path=/weights/xing4_0
   ```

2. **Draft model weights (only required for dspark mode, optional)**: If using dspark speculative decoding, verify the draft model directory exists and is readable:

   ```bash
   ls /hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_8_8/checkpoints/checkpoint_best
   ```

   > The mtp mode uses the MTP (Multi-Token Prediction) prediction head built into the main model, so no separate draft model weights are needed.

---

## 6. Starting the Inference Service

The service supports two speculative decoding modes, **either one** can be chosen; both can also be omitted (removing `--speculative-config` disables speculative decoding — the service still infers normally, just with slightly lower throughput):

| Mode | Speculative decoding method | Draft model required | Listening port | Log file |
| --- | --- | --- | --- | --- |
| Mode 1 | dspark (optional) | Yes | `8000` | `dspark-baseline-think.txt` |
| Mode 2 | mtp | No (uses the main model's MTP prediction head) | `8009` | `mtp-9.txt` |

### 6.1 Mode 1: dspark Speculative Decoding (Optional)

> **dspark is optional**: it depends on an additional draft model. If not enabled, the service still runs normally, just with slightly lower throughput. If enabled, make sure the draft model weights are ready (see Section 5).

```bash
nohup python -m vllm.entrypoints.openai.api_server \
    --model $weight_path \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name xingchen4 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 7, "method": "dspark", "model":"/hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_8_8/checkpoints/checkpoint_best"}' \
    --compilation-config '{"cudagraph_capture_sizes": [1,3,6,9,18,36,65,129,257,384,513,1026,2049],"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": false, "multistream_overlap_shared_expert": true, "multistream_overlap_gate": true}' \
    > dspark-baseline-think.txt 2>&1 &
```

> ⚠️ Note: In the original script, there is a stray `i` at the end of the `--model $weight_path` line — a typo, please remove it or argument parsing will fail.

### 6.2 Mode 2: mtp Speculative Decoding

> mtp (Multi-Token Prediction) uses the MTP prediction head of the main model itself; a single forward pass can predict multiple tokens, **no additional draft model is needed**, and no `model` field is required in `--speculative-config`.

```bash
nohup python -m vllm.entrypoints.openai.api_server \
    --model $weight_path \
    --port 8009 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name xingchen4 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.6 \
    --enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 2, "method": "mtp"}' \
    --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": false, "multistream_overlap_shared_expert": true, "multistream_overlap_gate": true}' \
    --compilation-config '{"cudagraph_capture_sizes": [1,3,6,9,18,36,65,129,257,384,513,1026,2049],"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    > mtp-9.txt 2>&1 &
```

### 6.3 Parameter Differences Between the Two Modes

| Parameter | Mode 1 (dspark) | Mode 2 (mtp) |
| --- | --- | --- |
| `--port` | `8000` | `8009` |
| `--tensor-parallel-size` | `2` | `2` |
| `--max-num-batched-tokens` | `8192` | `4096` |
| `--gpu-memory-utilization` | `0.85` | `0.6` |
| `--speculative-config` | `method=dspark`, `num_speculative_tokens=7`, includes draft model `model` path | `method=mtp`, `num_speculative_tokens=2`, no draft model |
| Draft model | Required | Not required |
| Log file | `dspark-baseline-think.txt` | `mtp-9.txt` |

All other parameters (model, service name, context length, prefix caching, graph capture and additional configs) are identical between the two modes.

### 6.4 Viewing Startup Logs

```bash
# dspark mode
tail -f dspark-baseline-think.txt
# mtp mode
tail -f mtp-9.txt
```

When the log shows something like `Application startup complete` or the listening port information, the service is ready.

---

## 7. Startup Parameter Explanation

### 7.1 Basic Parameters

| Parameter | Value | Description |
| --- | --- | --- |
| `--model` | `$weight_path` | Path to the main model weights directory |
| `--port` | `8000` (dspark) / `8009` (mtp) | Service listening port; must correspond to the selected startup mode |
| `--served-model-name` | `xingchen4` | Exposed model name; the `model` field in API request bodies must match this |
| `--trust-remote-code` | - | Allow loading custom code from the model directory |
| `--max-num-seqs` | `64` | Maximum number of concurrent sequences (requests) |
| `--max-model-len` | `131072` | Maximum context length, i.e. 128K tokens |
| `--max-num-batched-tokens` | `8192` (dspark) / `4096` (mtp) | Maximum number of tokens per batch; controls memory usage and throughput |

### 7.2 Parallelism and Resource Parameters

| Parameter | Value | Description |
| --- | --- | --- |
| `--tensor-parallel-size` | `2` | Use 2 NPUs for tensor parallelism (official parameter name; `--tensor-parallel` is a compatible alias with the same effect) |
| `--data-parallel-size` | `1` | Data parallelism degree; 1 for single-replica deployments |
| `--gpu-memory-utilization` | `0.85` (dspark) / `0.6` (mtp) | Upper limit of NPU memory utilization; lower values are more conservative and help prevent OOM |

### 7.3 Acceleration Feature Parameters

| Parameter | Value | Description |
| --- | --- | --- |
| `--enable-prefix-caching` | - | Enable prefix caching so requests with the same prefix can reuse the KV cache, improving throughput |
| `--speculative-config` | see scripts (optional) | Speculative decoding config, **choose one**: ① dspark: `method=dspark`, `num_speculative_tokens=7`, `model` pointing to the draft model path; ② mtp: `method=mtp`, `num_speculative_tokens=2`, no draft model. **Omitting this parameter disables speculative decoding** |
| `--compilation-config` | see scripts | CUDA Graph capture config; `FULL_AND_PIECEWISE` mode with multiple capture sizes reduces operator scheduling overhead |
| `--additional-config` | see scripts | Ascend/model extra config: `enable_cpu_binding` (CPU core binding), disable DSA CP, enable shared-expert multistream overlap and gate multistream overlap |

---

## 8. Service Testing

### 8.1 Health Check

```bash
curl http://localhost:8000/health    # dspark mode (port 8000)
curl http://localhost:8009/health    # mtp mode (port 8009)
```

Expected response: `OK`.

### 8.2 API Call (Completions)

```bash
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "xingchen4",
        "prompt": "Hello, please introduce you",
        "max_tokens": 1024,
        "temperature": 0,
        "seed": 1234
    }'
```

> ⚠️ **Port consistency**: The test port must match the service's actual listening port — dspark mode is `8000`, mtp mode is `8009`. The example below uses `8000`; replace it with `8009` when using mtp mode.

Request body field description:

| Field | Value | Description |
| --- | --- | --- |
| `model` | `xingchen4` | Must match `--served-model-name` |
| `prompt` | text | Input prompt |
| `max_tokens` | `1024` | Maximum number of tokens to generate |
| `temperature` | `0` | Sampling temperature; 0 means greedy decoding |
| `seed` | `1234` | Random seed for reproducible results |

### 8.3 Chat Endpoint (Optional)

The service also supports the OpenAI Chat format:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "xingchen4",
        "messages": [{"role": "user", "content": "Hello, please introduce you"}],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

---

## 9. mtp Mode Test Baseline

The following is the benchmark baseline for **mtp mode** (port `8009`, see 6.2) measured on Ascend 910B4; latency metrics in the table are all in **ms**. `run` is the round number, `avg` is the average row:

| run | TTFT_avg | TTFT_p90 | TTFT_max | TPOT_avg | TPOT_p90 | TPOT_max | ITL_p90 | ITL_p99 | Spec% | gen/s | E2E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 132.15 | 142.43 | 142.43 | 8.64 | 10.14 | 10.14 | 29.55 | 36.30 | 69.42 | 92.02 | 710.27 |
| 2 | 130.84 | 142.76 | 142.76 | 8.73 | 10.12 | 10.12 | 29.30 | 55.53 | 69.42 | 91.49 | 709.48 |
| 3 | 145.70 | 169.62 | 169.62 | 8.77 | 10.13 | 10.13 | 29.30 | 56.26 | 69.42 | 89.00 | 736.90 |
| avg | 136.23 | 151.60 | 151.60 | 8.71 | 10.13 | 10.13 | 29.38 | 49.36 | 69.42 | 90.84 | 718.88 |

Field description:

| Field | Meaning |
| --- | --- |
| `run` | Round number (1/2/3); `avg` is the average row |
| `TTFT_avg / TTFT_p90 / TTFT_max` | Time to first token (ms): average / P90 / maximum |
| `TPOT_avg / TPOT_p90 / TPOT_max` | Time per output token (ms): average / P90 / maximum |
| `ITL_p90 / ITL_p99` | Inter-token latency (ms): P90 / P99 |
| `Spec%` | Speculative decoding acceptance rate (stable at 69.42% in this group) |
| `gen/s` | Generation rate (tokens/s) |
| `E2E` | End-to-end latency (ms) |

---

## 10. Appendix

### Appendix A: Container Startup Script `run_container.sh`

```bash
#!/bin/bash
# Usage: bash run_container.sh <container_name> <image_name>
CONTAINER_NAME=$1
IMAGE=$2

docker run -itd -u 0 \
    --name $CONTAINER_NAME \
    --net=host \
    --privileged=true \
    --shm-size=512g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /mnt/sfs_turbo/.cache:/root/.cache \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /var/log/npu/:/usr/slog \
    -e VLLM_USE_V1=1 \
    -it $IMAGE /bin/bash
```

Usage example:

```bash
bash run_container.sh xing4_0-vllm harbor.telecom-ai.com.cn/library/vllm-ascend:v0.26.0rc1-29b-xing4_0
```

### 10.2 dspark Startup Script `start_dspark.sh` (Optional)

```bash
#!/bin/bash
export weight_path=/models/xing4_0   # modify to match the actual weight path

nohup python -m vllm.entrypoints.openai.api_server \
    --model $weight_path \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name xingchen4 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 7, "method": "dspark", "model":"/hpfs/huawei-2607/shangxt/Telecom-29B/telecom_80w_no_thinking_0910_8_8/checkpoints/checkpoint_best"}' \
    --compilation-config '{"cudagraph_capture_sizes": [1,3,6,9,18,36,65,129,257,384,513,1026,2049],"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": false, "multistream_overlap_shared_expert": true, "multistream_overlap_gate": true}' \
    > dspark-baseline-think.txt 2>&1 &
```

### Appendix C: mtp Startup Script `start_mtp.sh`

```bash
#!/bin/bash
export weight_path=/models/xing4_0   # modify to match the actual weight path

nohup python -m vllm.entrypoints.openai.api_server \
    --model $weight_path \
    --port 8009 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name xingchen4 \
    --max-num-seqs 64 \
    --max-model-len 131072 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.6 \
    --enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 2, "method": "mtp"}' \
    --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": false, "multistream_overlap_shared_expert": true, "multistream_overlap_gate": true}' \
    --compilation-config '{"cudagraph_capture_sizes": [1,3,6,9,18,36,65,129,257,384,513,1026,2049],"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    > mtp-9.txt 2>&1 &
```

### Appendix D: Test Script `test_api.sh`

```bash
#!/bin/bash
# The port must match the --port used at startup: dspark=8000, mtp=8009
PORT=${PORT:-8000}
curl -X POST http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "xingchen4",
        "prompt": "Hello, please introduce you",
        "max_tokens": 1024,
        "temperature": 0,
        "seed": 1234
    }'
```
