# Deploying Xing4.0-29B-A4B with the vLLM Plugin (Open-Source Path)

　　This document describes how to deploy an OpenAI-compatible API service for the Xing4.0-29B-A4B model through the vLLM **open-source plugin (general plugin)** mechanism, using the official vLLM image as the base environment, without modifying the vLLM source tree. It covers:

- Prerequisites (official vLLM image)
- Downloading the Xing4.0-29B-A4B model
- Getting the plugin and starting the container
- Launching the vLLM service
- Verifying plugin loading and its switches
- OpenAI-compatible API call examples
- Startup parameter descriptions and inference parameter recommendations
- Troubleshooting

　　The vLLM adaptation code for Xing4.0 is released as an **open-source plugin**, hosted at [XingChen-AGI/Xing4.0-plugin](https://github.com/XingChen-AGI/Xing4.0-plugin). The plugin is distributed as a Python package and registers itself through a Python entry point in the `vllm.general_plugins` group, so vLLM loads it automatically at process startup. It works with the official vLLM image, requires no changes to the vLLM source tree, and can be upgraded and maintained independently. The plugin **contains no custom CUDA kernels and needs no separate build**; its implementation reuses runtime operators already present in vLLM (including `vllm.model_executor.layers.mhc`).

> [!NOTE]
> The plugin registers: the `Xing4_0ForCausalLM` model implementation, the `xing4_0` model configuration (reusing vLLM's MLA + MoE configuration infrastructure), the `xing4_0` reasoning parser, the `xing4_0` tool-call parser, and `xing4_0` speculative decoding (MTP) plus MLA detection support.

## Prerequisites

　　Make sure the host machine meets the following software requirements:

- Docker (24.0 or later recommended)
- NVIDIA driver (535 or later recommended)
- NVIDIA Container Toolkit (the `nvidia-docker` runtime)
- A GPU that supports bfloat16 (mHC mode requires bfloat16)

　　This document uses the official vLLM image `vllm/vllm-openai` as the base environment: **there is no need to install vLLM on the host, nor to build vLLM yourself** — simply install this plugin inside the container to enable Xing4.0 support.

　　To check that the GPU is accessible from a container:

```bash
docker run --rm --gpus all vllm/vllm-openai:latest nvidia-smi
```

　　Other dependencies required inside the container (such as tilelang, needed by the mHC path on CUDA) are usually installed together with vLLM in the image; if missing, run `pip install tilelang` inside the container.

> [!NOTE]
> The image tag can be pinned to a specific vLLM version as needed. The plugin is ported from the upstream vLLM commit `2e09247c2d7b6b97d13af6e71a85bf8d1271deb6`, which is recorded in the `TESTED_VLLM` constant in `src/vllm_xing4_0_plugin/__init__.py`; for production, pin the version rather than using `latest`.

> [!NOTE]
> The commands in this document use `--tensor-parallel-size 2` as an example, which requires at least 2 GPUs; adjust this parameter according to the actual number of GPUs available. The NVIDIA H100 used in this document (80 GB of memory per card, 160 GB across 2 cards) is only the **hardware environment used for demonstration and validation**; it is not a mandatory requirement.

## Download the Xing4.0-29B-A4B Model

　　The Xing4.0-29B-A4B model can be downloaded from the following platforms; the weights are identical across all three:

| Platform | Address |
|----------|---------|
| Hugging Face | https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B |
| ModelScope | https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B |
| Modelers | https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B |

　　Taking ModelScope as an example, download to a local directory:

```bash
pip install modelscope

modelscope download \
  --model XingChen-AGI/Xing4.0-29B-A4B \
  --local_dir /yourpath/models/Xing4.0-29B-A4B
```

　　To further reduce memory usage, you can also use the FP8 version ([Hugging Face](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8) | [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.0-29B-A4B-FP8)); the download and startup steps are the same as below, you only need to point the mounted path to the FP8 directory.

> [!NOTE]
> mHC mode (`hc_mult > 1`) supports `bfloat16` only and does not support pipeline parallelism. The plugin reports a clear error when these conditions are not met.

## Getting the Plugin and Starting the Container

　　Step 1: get the plugin source code, so that it can be mounted into the container:

```bash
git clone https://github.com/XingChen-AGI/Xing4.0-plugin.git
```

　　Step 2: start a container (with bash as the entry point, kept running), mounting the model directory and the plugin repository and exposing the API port:

```bash
docker run -d \
  --name xing4-vllm-plugin \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  -v /yourpath/models/Xing4.0-29B-A4B:/models/Xing4.0-29B-A4B \
  -v /yourpath/Xing4.0-plugin:/workspace/Xing4.0-plugin \
  vllm/vllm-openai:latest \
  -c "sleep infinity"
```

　　Step 3: enter the container and install the plugin inside it (the vLLM environment is already present there):

```bash
docker exec -it xing4-vllm-plugin bash

pip install -e /workspace/Xing4.0-plugin --no-deps
```

　　Step 4: verify that the entry point is visible to the interpreter inside the container:

```bash
python -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='vllm.general_plugins')])"
```

　　The output should include:

```
('xing4_0', 'vllm_xing4_0_plugin:register')
```

> [!NOTE]
> Adding `src/` to `PYTHONPATH` alone does **not** make the plugin discoverable, because the entry point comes from the metadata of an installed package. The installation step above is required. `--no-deps` prevents the plugin from replacing the vLLM installation inside the container.

## Launch the vLLM Service

　　Run the following command inside the container to start the service — no additional plugin-specific arguments are needed:

```bash
vllm serve /models/Xing4.0-29B-A4B \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Xing4.0-29B-A4B \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 4 \
    --reasoning-parser xing4_0 \
    --tool-call-parser xing4_0 \
    --enable-auto-tool-choice \
    --speculative-config '{"method":"mtp", "num_speculative_tokens": 1}'
```

　　Once launched, the OpenAI-compatible API is available at `http://localhost:8000/v1`. The current terminal keeps printing service logs; once a line such as `Uvicorn running on http://0.0.0.0:8000` appears, the service is available externally.

> [!NOTE]
> The command above enables MTP speculative decoding (available when the model repository ships MTP layers); drop `--speculative-config` if you do not need it. `mtp` is vLLM's speculative decoding method name for MTP-style draft models; when `model` is not specified, vLLM automatically reuses the MTP layers in the target model weights, and the plugin already maps the `xing4_0` MTP draft configuration onto that path.

### Verifying Plugin Loading

　　At service startup, confirm the plugin's registration record in the logs:

```
vllm-xing4_0-plugin: registered Xing4_0ForCausalLM -> vllm_xing4_0_plugin.xing4_0:Xing4_0ForCausalLM (tested against vLLM commit 2e09247c2d7b6b97d13af6e71a85bf8d1271deb6, running <vLLM version>)
```

　　If the runtime environment already ships Xing4_0 support (such as the prebuilt image `quay.io/xingchen-agi/xingchen-inference-vllm:v0.29.1rc1-xing4_0`), the plugin logs a stand-down message and keeps the in-environment implementation active:

```
vllm-xing4_0-plugin: Xing4_0ForCausalLM is already supported in tree; standing down. Set XING4_0_PLUGIN_OVERRIDE=1 to override.
```

　　To explicitly use this plugin (ported) implementation, set the environment variable before starting:

```bash
XING4_0_PLUGIN_OVERRIDE=1 vllm serve /models/Xing4.0-29B-A4B ...
```

　　If your environment uses the `VLLM_PLUGINS` allowlist, add this plugin's entry point name `xing4_0` to it, otherwise the plugin will not be loaded:

```bash
VLLM_PLUGINS=xing4_0 vllm serve /models/Xing4.0-29B-A4B ...
```

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

　　To disable chain-of-thought output, pass `"chat_template_kwargs":{"enable_thinking":false}` in the request body. With thinking enabled, the generated content looks like `<think>reasoning process</think>final content`; with thinking disabled, only the final content is produced.

## Startup Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| `--host 0.0.0.0` | Listen on all network interfaces so the service is reachable from outside the container |
| `--port 8000` | API service port |
| `--served-model-name` | The model name exposed externally; API calls must use the same name |
| `--tensor-parallel-size 2` | Number of GPUs for tensor parallelism; must match the number of available GPUs |
| `--trust-remote-code` | Trust remote code shipped in the model repository |
| `--max-model-len 262144` | Maximum context length; Xing4.0 natively supports 256K |
| `--gpu-memory-utilization 0.90` | Upper limit of the fraction of GPU memory to reserve; adjust according to available headroom |
| `--max-num-seqs 4` | Maximum number of concurrent requests; adjust according to traffic and memory |
| `--reasoning-parser xing4_0` | Parses Xing4.0 chain-of-thought output (registered by the plugin) |
| `--tool-call-parser xing4_0` | Parses the Xing4.0 tool-call format (registered by the plugin) |
| `--enable-auto-tool-choice` | Enables automatic tool calling (function calling); required when requests carry a `tools` parameter |
| `--speculative-config` | Enables MTP speculative decoding to accelerate generation; use `mtp` as the `method` |
| `--chat-template` | Optional: path to a `chat_template.jinja` outside the model repository |

## Recommended Inference Parameters

　　Following Xing4.0's official recommendations, the sampling parameters for different scenarios are:

| Scenario | temperature | top_p | repetition_penalty |
|----------|-------------|-------|---------------------|
| Complex reasoning / general tasks | 1.0 | 0.95 | 1.05 |
| Coding / agent tasks | 0.8 | 0.95 | 1.05 |

## Troubleshooting

### The plugin was not loaded

- Make sure the plugin is installed in the **same Python environment as vLLM inside the container**, and that you completed a package installation rather than only setting `PYTHONPATH`
- Use the command in "Getting the Plugin and Starting the Container" to check that the entry point is visible; the output should include `('xing4_0', 'vllm_xing4_0_plugin:register')`
- If the `VLLM_PLUGINS` allowlist is set, it must include `xing4_0`
- If `vllm-xing4_0-plugin: registered ...` does not appear in the logs, first check whether the stand-down logic was triggered (see "Verifying Plugin Loading")

### The model architecture is not recognized (Xing4_0ForCausalLM missing)

- If the plugin did not register successfully, vLLM refuses to load the model because the architecture is missing. Follow the previous item to check registration and logs
- In environments that already ship support (such as the prebuilt image), you can set `XING4_0_PLUGIN_OVERRIDE=1` to force the plugin implementation instead

### tilelang is reported as missing

- The mHC path on CUDA requires tilelang; run `pip install tilelang` inside the container and restart the service

### mHC-related errors (bfloat16 / pipeline parallelism)

- mHC mode requires both the model and inference to run in `bfloat16`; FP16 or quantized paths raise errors
- mHC does not currently support pipeline parallelism; do not set `--pipeline-parallel-size > 1`

### Out of memory (OOM)

- Lower `--gpu-memory-utilization` (e.g. 0.85) or `--max-model-len` (e.g. 131072)
- Use the FP8 version of the model to reduce memory usage
- Reduce `--max-num-seqs` to lower the peak memory usage under concurrency

### Tensor parallelism does not match the GPU count

- `--tensor-parallel-size` must divide the number of visible GPUs; for example, with 4 GPUs it can be set to `4` or `2`

### The container cannot access the GPU

- Make sure the NVIDIA Container Toolkit is installed and start the container with `--gpus all`
- To use only a subset of the GPUs, use `--gpus '"device=0,1"'` instead

### The service is not reachable externally

- Make sure `--host` is set to `0.0.0.0` and the port is published with `-p 8000:8000`

## Upgrade and Maintenance

　　The plugin depends on vLLM's internal runtime APIs, and the `TESTED_VLLM` constant records the vLLM version the port is based on. After upgrading vLLM (i.e. switching the image tag):

- Re-run the plugin's own smoke tests (no GPU or model weights required) from the plugin repository inside the container, at `/workspace/Xing4.0-plugin`, to confirm that the registration entry point, the MTP/MLA patches, and the parsers work as expected:

```bash
python -m pytest tests/test_smoke.py -q
```

- Check the "ported baseline version" and the "running vLLM version" printed by the plugin in the startup logs; if the two diverge significantly, watch for internal API migrations and re-port the relevant modules when necessary
