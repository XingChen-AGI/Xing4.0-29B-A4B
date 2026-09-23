# Deploying the Xing4.0-29B-A4B Model with vLLM

　　This document describes how to launch an OpenAI-compatible API service for the Xing4.0-29B-A4B model with vLLM. It covers:

- Pulling the prebuilt image
- Downloading the Xing4.0-29B-A4B model
- Starting the vLLM service with a single Docker command
- OpenAI-compatible API call examples
- Startup parameter descriptions and inference parameter recommendations

　　The vLLM adaptation code for Xing4.0 ([vllm-project/vllm#57135](https://github.com/vllm-project/vllm/pull/57135)) has not yet been merged into the vLLM main branch. Until it is merged, this document uses the prebuilt image `quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0`, which already integrates this PR, so you can deploy quickly without building from source.

## Prerequisites

　　Make sure the host machine meets the following software requirements:

- Docker (24.0 or later recommended)
- NVIDIA driver (535 or later recommended)
- NVIDIA Container Toolkit (the `nvidia-docker` runtime)

　　Verify that the GPU is accessible from a container:

```bash
docker run --rm --gpus all quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0 nvidia-smi
```

　　The startup commands in this document use `--tensor-parallel-size 2` as an example, which requires at least 2 GPUs; adjust this parameter according to the actual number of GPUs available.

> [!NOTE]
> The NVIDIA H100 used in this document (80 GB of memory per card, 160 GB across 2 cards) is only the **hardware environment used for demonstration and validation**; it is not a mandatory requirement. You may use other CUDA-capable NVIDIA GPUs according to your own hardware — simply adjust parameters such as `--tensor-parallel-size`, `--max-model-len`, and `--gpu-memory-utilization` based on the actual memory and number of cards. If memory is limited, you can also use the FP8 version of the model as described below.

## Pull the Image

```bash
docker pull quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0
```

## Download the Xing4.0-29B-A4B Model

　　The Xing4.0-29B-A4B model can be downloaded from the following platforms; the weights are identical across all three:

| Platform | URL |
|----------|-----|
| Hugging Face | https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B |
| ModelScope | https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B |
| Modelers | https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B |

　　Using ModelScope as an example, download it to a local directory:

```bash
pip install modelscope

modelscope download \
  --model XingChen-AGI/Xing4.0-29B-A4B \
  --local_dir /yourpath/models/Xing4.0-29B-A4B
```

　　To further reduce GPU memory usage, you can also use the FP8 version ([Hugging Face](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8)). The download and startup procedures are the same as below; simply point `MODEL_PATH` to the FP8 directory.

## Start the vLLM Service

　　Start the vLLM service with Docker, mounting the local model directory into the container and exposing the API port:

```bash
docker run -d \
  --name xing4-vllm \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v /yourpath/models/Xing4.0-29B-A4B:/models/Xing4.0-29B-A4B \
  quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0 \
  vllm serve /models/Xing4.0-29B-A4B \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Xing4.0-29B-A4B \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --reasoning-parser xing4 \
    --tool-call-parser xing4 \
    --speculative-config '{"method":"mtp", "num_speculative_tokens": 1}'
```

　　After startup, the OpenAI-compatible API is available at `http://localhost:8000/v1`. Check the startup logs to confirm that the service is ready:

```bash
docker logs -f xing4-vllm
```

　　When a message similar to `Uvicorn running on http://0.0.0.0:8000` appears in the logs, the service is ready to accept requests.

### Startup Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| `--host 0.0.0.0` | Listen on all network interfaces so the service is accessible from outside the container |
| `--port 8000` | API service port |
| `--served-model-name` | The model name exposed externally; API calls must use this name |
| `--tensor-parallel-size 2` | Number of GPUs for tensor parallelism; must match the number of available GPUs |
| `--trust-remote-code` | Trust remote code shipped inside the model repository |
| `--max-model-len 262144` | Maximum context length; Xing4.0 natively supports 256K |
| `--gpu-memory-utilization 0.90` | Upper limit of the fraction of GPU memory to reserve; adjust according to available headroom |
| `--max-num-seqs 32` | Maximum number of concurrent requests |
| `--reasoning-parser xing4` | Parses the chain-of-thought output of Xing4.0 (the reasoning process comes before `</think>`) |
| `--tool-call-parser xing4` | Parses the tool-call format of Xing4.0 |
| `--speculative-config` | Enables MTP speculative decoding to accelerate generation |

## Calling the API

　　Once the service is running, you can call it through the OpenAI-compatible interface. The `model` field in the examples below must match the `--served-model-name` used at startup.

### Basic Chat (curl)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Xing4.0-29B-A4B",
    "messages": [
      {"role": "user", "content": "Introduce yourself in one sentence."}
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "chat_template_kwargs":{"enable_thinking":true},"skip_special_tokens": false
  }'
```

　　Because the `xing4` reasoning parser is enabled, `message.reasoning_content` in the response contains the chain-of-thought content, while `message.content` contains the final answer.

## Inference Parameter Recommendations

　　Based on the official Xing4.0 recommendations, the sampling parameters for different scenarios are as follows:

| Scenario | temperature | top_p | repetition_penalty |
|----------|-------------|-------|---------------------|
| Complex reasoning / general tasks | 1.0 | 0.95 | 1.05 |
| Coding / agent tasks | 0.8 | 0.95 | 1.05 |

## FAQ

### Out of Memory (OOM)

- Lower `--gpu-memory-utilization` (e.g. 0.85) or `--max-model-len` (e.g. 131072)
- Use the FP8 version of the model to reduce GPU memory usage
- Reduce `--max-num-seqs` to lower the peak memory usage under concurrency

### Tensor Parallel Size Mismatch

- `--tensor-parallel-size` must divide the number of visible GPUs; for example, with 4 GPUs it can be set to `4` or `2`

### GPU Not Accessible Inside the Container

- Make sure the NVIDIA Container Toolkit is installed and that `--gpus all` is used at startup
- If you only want to use specific GPUs, use `--gpus '"device=0,1"'` instead

### Service Not Accessible Externally

- Make sure `--host` is set to `0.0.0.0` and that the port is mapped with `-p 8000:8000`
